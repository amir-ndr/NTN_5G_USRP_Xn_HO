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

Delay modes (--delay):
    normal  : software time.sleep() after packet arrival  [default]
    NE-ONE  : real hair-pin through NE-ONE emulator
              Flow: USRP RX → UDP send (10.10.2.1 → 10.10.2.2) → NE-ONE adds GND delay
                    → UDP recv (10.10.2.2:5102) → log & display
              Requires Linux with eth3/eth4 wired to NE-ONE and ip routes configured.
              controller.py must also run with --delay=NE-ONE to push delay values.

Usage:
    python3 TN.py --addr 192.168.10.5
    python3 TN.py --addr 192.168.10.5 --gain 25 --no-plot
    python3 TN.py --addr 192.168.10.5 --no-ext-ref
    python3 TN.py --addr 192.168.10.5 --delay NE-ONE
"""

import argparse
import csv
import json
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from radio import USRP, Visualizer

NODE_NAME   = "TN"
_HERE       = Path(__file__).resolve().parent
DELAYS_FILE = _HERE / "delays.json"
LOG_FILE    = _HERE / "TN_log.csv"
LOG_HEADER  = ["timestamp", "seq", "msg_id", "prop_delay_ms", "gnd_ms"]

# ── NE-ONE hair-pin addresses (only used with --delay=NE-ONE) ─────────────────
# Must match what you configured with `ip addr add` on the Linux machine:
#   sudo ip addr add 10.10.2.1/30 dev eth3   (send side  → into NE-ONE)
#   sudo ip addr add 10.10.2.2/30 dev eth4   (recv side  ← out of NE-ONE)
_NEONE_SEND_IP      = "10.10.2.1"   # eth3 — exit toward NE-ONE
_NEONE_RECV_IP      = "10.10.2.2"   # eth4 — return from NE-ONE (after GND delay)
_NEONE_HAIRPIN_PORT = 5102           # arbitrary UDP port for the GND hair-pin
# ─────────────────────────────────────────────────────────────────────────────


def _read_gnd_delay_s() -> tuple[float, float]:
    """Returns (delay_s, gnd_ms). Both 0.0 if file not ready."""
    try:
        with open(DELAYS_FILE) as f:
            d = json.load(f)
        gnd_ms = d.get("gnd_ms", 0.0)
        return gnd_ms / 1000.0, gnd_ms
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 0.0, 0.0


def print_packet(pkt_dict: dict, delay_ms: float, via_neone: bool = False):
    tag = "  [NE-ONE]" if via_neone else ""
    print(f"\n{'─'*50}")
    print(f"  [{NODE_NAME}]{tag}  seq={pkt_dict['seq']:03d}  msg_id={pkt_dict['msg_id']}  "
          f"prop_delay={delay_ms:.2f} ms")
    print(f"{'─'*50}")
    for key, val in pkt_dict.get("payload", {}).items():
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


# ── NE-ONE receiver thread ────────────────────────────────────────────────────

def _neone_recv_loop(
    records:    list,
    writer:     csv.DictWriter,
    csvfile,
    stop_event: threading.Event,
) -> None:
    """
    Background thread — NE-ONE mode only.

    Listens on eth4 (10.10.2.2:5102) for JSON-encoded packets that have
    already been delayed by NE-ONE on their way back from eth3.
    Processes and logs each packet exactly as normal mode would after a sleep.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((_NEONE_RECV_IP, _NEONE_HAIRPIN_PORT))
    sock.settimeout(0.5)
    print(f"[{NODE_NAME}][NE-ONE] Hairpin receiver ready  "
          f"{_NEONE_RECV_IP}:{_NEONE_HAIRPIN_PORT}", flush=True)

    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue

        try:
            d = json.loads(data.decode())
        except Exception:
            continue

        # Read current GND delay for logging (real delay was applied by NE-ONE)
        delay_s, gnd_ms = _read_gnd_delay_s()
        delay_ms = delay_s * 1000.0

        print_packet(d, delay_ms, via_neone=True)

        row = {
            "timestamp":     datetime.now(tz=timezone.utc).isoformat(),
            "seq":           d.get("seq", -1),
            "msg_id":        d.get("msg_id", "0x00"),
            "prop_delay_ms": round(delay_ms, 3),
            "gnd_ms":        round(gnd_ms, 3),
        }
        writer.writerow(row)
        csvfile.flush()
        records.append(row)

    sock.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="USRP X310 BPSK Receiver — TN ground node")
    p.add_argument("--addr",       default="192.168.10.5",  help="RX USRP IP address")
    p.add_argument("--gain",       type=float, default=20.0, help="RX gain in dB")
    p.add_argument("--no-ext-ref", action="store_true",      help="Use internal clock (no OctoClock-G)")
    p.add_argument("--no-plot",    action="store_true",      help="Disable live signal plot")
    p.add_argument("--delay",      choices=["normal", "NE-ONE"], default="normal",
                   help="Delay mode: 'normal'=software sleep (default), "
                        "'NE-ONE'=real hair-pin through NE-ONE emulator")
    args = p.parse_args()

    viz = None if args.no_plot else Visualizer(title="TN — Signal Monitor", position="right")

    records    = []
    stop_event = threading.Event()

    with open(LOG_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=LOG_HEADER)
        writer.writeheader()
        csvfile.flush()

        # ── NE-ONE: start background receiver thread ──────────────────────────
        neone_thread = None
        if args.delay == "NE-ONE":
            neone_thread = threading.Thread(
                target=_neone_recv_loop,
                args=(records, writer, csvfile, stop_event),
                daemon=True,
                name="neone-gnd-recv",
            )
            neone_thread.start()

        with USRP(addr=args.addr, role="rx", gain=args.gain, ext_ref=not args.no_ext_ref) as radio:
            print(f"\n[{NODE_NAME}] Listening — Ctrl+C to stop\n")
            print(f"[{NODE_NAME}] Logging to {LOG_FILE}\n")
            if args.delay == "NE-ONE":
                print(f"[{NODE_NAME}] Delay mode : NE-ONE  "
                      f"(USRP RX → UDP → {_NEONE_SEND_IP} → NE-ONE → "
                      f"{_NEONE_RECV_IP}:{_NEONE_HAIRPIN_PORT})\n")
            else:
                print(f"[{NODE_NAME}] Delay mode : normal  (software sleep from delays.json)\n")

            try:
                while True:
                    pkt, info = radio.receive()

                    if viz:
                        viz.update(info, pkt.payload.get("data", "") if pkt else "")

                    if pkt:
                        dest = pkt.payload.get("dest", "")
                        if dest == NODE_NAME:
                            if args.delay == "NE-ONE":
                                # Serialize and inject into the NE-ONE hair-pin.
                                # The packet exits eth3, NE-ONE adds GND delay,
                                # then it arrives at the receiver thread on eth4.
                                payload = json.dumps({
                                    "seq":     pkt.seq,
                                    "msg_id":  f"{pkt.msg_id:#04x}",
                                    "payload": pkt.payload,
                                }).encode()
                                tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                                tx.bind((_NEONE_SEND_IP, 0))
                                tx.sendto(payload, (_NEONE_RECV_IP, _NEONE_HAIRPIN_PORT))
                                tx.close()
                                # Processing happens in _neone_recv_loop after delay
                            else:
                                # Normal mode: software sleep then process inline
                                delay_s, gnd_ms = _read_gnd_delay_s()
                                if delay_s > 0:
                                    time.sleep(delay_s)
                                pkt_dict = {
                                    "seq":     pkt.seq,
                                    "msg_id":  f"{pkt.msg_id:#04x}",
                                    "payload": pkt.payload,
                                }
                                print_packet(pkt_dict, delay_s * 1000)
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
                            print(f"[{NODE_NAME}] seq={pkt.seq:03d}  dest={dest!r}  (ignored)",
                                  flush=True)

            except KeyboardInterrupt:
                print(f"\n[{NODE_NAME}] Stopped.")
                stop_event.set()
                if neone_thread:
                    neone_thread.join(timeout=2.0)
                print_summary(records)


if __name__ == "__main__":
    main()
