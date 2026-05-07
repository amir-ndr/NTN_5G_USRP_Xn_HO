#!/usr/bin/env python3
"""
TN.py — USRP X310 BPSK Receiver (terrestrial network / ground node)

Receives all packets over the air but only processes those addressed to "TN".
When a packet is addressed to this node, sleeps for the sat-to-ground one-way
propagation delay read from delays.json (written by controller.py from real
Starlink TLEs) before printing — emulating orbital propagation latency in software.

Packets addressed to "trgSAT" are received at RF but silently ignored at app layer —
the visualizer still updates so you can see RF activity regardless.

Each accepted packet is logged to TN_log.csv. A summary (packet count,
avg/min/max latency) is printed on Ctrl+C.

Usage:
    python3 TN.py --addr 192.168.10.5
    python3 TN.py --addr 192.168.10.5 --gain 25 --no-plot
    python3 TN.py --addr 192.168.10.5 --no-ext-ref
"""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from radio import USRP, Visualizer

NODE_NAME   = "TN"
_HERE       = Path(__file__).resolve().parent
DELAYS_FILE = _HERE / "delays.json"
LOG_FILE    = _HERE / "TN_log.csv"
LOG_HEADER  = ["timestamp", "seq", "msg_id", "prop_delay_ms", "gnd_ms"]


def _read_gnd_delay_s() -> tuple[float, float]:
    """Returns (delay_s, gnd_ms). Both 0.0 if file not ready."""
    try:
        with open(DELAYS_FILE) as f:
            d = json.load(f)
        gnd_ms = d.get("gnd_ms", 0.0)
        return gnd_ms / 1000.0, gnd_ms
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 0.0, 0.0


def print_packet(pkt, delay_s: float):
    print(f"\n{'─'*50}")
    print(f"  [{NODE_NAME}]  seq={pkt.seq:03d}  msg_id={pkt.msg_id:#04x}  "
          f"prop_delay={delay_s*1000:.2f} ms")
    print(f"{'─'*50}")
    for key, val in pkt.payload.items():
        print(f"  {key:<12} : {val}")
    print(f"{'─'*50}")


def print_summary(records: list[dict]):
    n = len(records)
    if n == 0:
        print(f"\n[{NODE_NAME}] No packets received.")
        return
    latencies = [r["prop_delay_ms"] for r in records]
    avg = sum(latencies) / n
    print(f"\n{'═'*50}")
    print(f"  [{NODE_NAME}] Session summary")
    print(f"{'═'*50}")
    print(f"  Packets received : {n}")
    print(f"  Avg latency      : {avg:.2f} ms")
    print(f"  Min latency      : {min(latencies):.2f} ms")
    print(f"  Max latency      : {max(latencies):.2f} ms")
    print(f"  Log saved to     : {LOG_FILE}")
    print(f"{'═'*50}\n")


def main():
    p = argparse.ArgumentParser(description="USRP X310 BPSK Receiver — TN ground node")
    p.add_argument("--addr",       default="192.168.10.5",  help="RX USRP IP address")
    p.add_argument("--gain",       type=float, default=20.0, help="RX gain in dB")
    p.add_argument("--no-ext-ref", action="store_true",      help="Use internal clock (no OctoClock-G)")
    p.add_argument("--no-plot",    action="store_true",      help="Disable live signal plot")
    args = p.parse_args()

    viz = None if args.no_plot else Visualizer(title="TN — Signal Monitor", position="right")

    records = []

    with open(LOG_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=LOG_HEADER)
        writer.writeheader()
        csvfile.flush()

        with USRP(addr=args.addr, role="rx", gain=args.gain, ext_ref=not args.no_ext_ref) as radio:
            print(f"\n[{NODE_NAME}] Listening — Ctrl+C to stop\n")
            print(f"[{NODE_NAME}] Logging to {LOG_FILE}\n")
            try:
                while True:
                    pkt, info = radio.receive()

                    if viz:
                        viz.update(info, pkt.payload.get("data", "") if pkt else "")

                    if pkt:
                        dest = pkt.payload.get("dest", "")
                        if dest == NODE_NAME:
                            delay_s, gnd_ms = _read_gnd_delay_s()
                            if delay_s > 0:
                                time.sleep(delay_s)
                            print_packet(pkt, delay_s)

                            row = {
                                "timestamp":     datetime.now(tz=timezone.utc).isoformat(),
                                "seq":           pkt.seq,
                                "msg_id":        f"{pkt.msg_id:#04x}",
                                "prop_delay_ms": round(delay_s * 1000, 3),
                                "gnd_ms":        round(gnd_ms, 3),
                            }
                            writer.writerow(row)
                            csvfile.flush()
                            records.append(row)
                        else:
                            print(f"[{NODE_NAME}] seq={pkt.seq:03d}  dest={dest!r}  (ignored)", flush=True)

            except KeyboardInterrupt:
                print(f"\n[{NODE_NAME}] Stopped.")
                print_summary(records)


if __name__ == "__main__":
    main()
