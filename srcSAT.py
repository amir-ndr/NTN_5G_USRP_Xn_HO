#!/usr/bin/env python3
"""
srcSAT.py — USRP X310 BPSK Transmitter (source satellite)

Sends structured packets over the air. The "dest" field in each packet
addresses either the target satellite (trgSAT) or the TN ground node (TN).

Currently: destination is chosen randomly each packet.
Later:      controller.py will write to dest_override.txt to steer routing
            based on the Xn HO algorithm — srcSAT will read that file.

Usage:
    python3 srcSAT.py --addr 192.168.20.6
    python3 srcSAT.py --addr 192.168.20.6 --gain 10
    python3 srcSAT.py --addr 192.168.20.6 --no-ext-ref
"""

import argparse
import json
import random
import time
from pathlib import Path
from radio import USRP

DESTINATIONS = ["trgSAT", "TN"]

DEST_OVERRIDE_FILE = "dest_override.txt"
DELAYS_FILE        = Path(__file__).resolve().parent / "delays.json"

BASE_MESSAGE = {
    "node_id":   "srcSAT",
    "freq_hz":   "3450000000",
    "mode":      "BPSK",
    "power_dbm": "10",
    "location":  "LEO-orbit",
    "status":    "active",
    "data":      "Hello from srcSAT",
}


def _read_dest_override() -> str | None:
    """Return destination from override file, or None if not set / file missing."""
    try:
        txt = open(DEST_OVERRIDE_FILE).read().strip()
        if txt in DESTINATIONS:
            return txt
    except FileNotFoundError:
        pass
    return None


def _read_task_type() -> str:
    """Read the current task type written by controller.py into delays.json."""
    try:
        with open(DELAYS_FILE) as f:
            return json.load(f).get("task_type", "mixed")
    except (FileNotFoundError, json.JSONDecodeError):
        return "mixed"


def main():
    p = argparse.ArgumentParser(description="USRP X310 BPSK Transmitter — srcSAT")
    p.add_argument("--addr",       default="192.168.20.6",  help="TX USRP IP address")
    p.add_argument("--gain",       type=float, default=10.0, help="TX gain in dB")
    p.add_argument("--no-ext-ref", action="store_true",      help="Use internal clock (no OctoClock-G)")
    p.add_argument("--dest",       choices=DESTINATIONS,     help="Force a fixed destination (overrides random + file)")
    args = p.parse_args()

    with USRP(addr=args.addr, role="tx", gain=args.gain, ext_ref=not args.no_ext_ref) as radio:
        print("\nTransmitting continuously — Ctrl+C to stop\n")
        pkt = None
        try:
            while True:
                # Destination priority: CLI flag > override file > random
                if args.dest:
                    dest = args.dest
                else:
                    dest      = _read_dest_override() or random.choice(DESTINATIONS)
                task_type = _read_task_type()

                msg = {**BASE_MESSAGE, "dest": dest, "task_type": task_type}
                pkt = radio.transmit(msg, msg_id=0x01)
                print(f"[srcSAT] seq={pkt.seq:03d}  dest={dest}  task={task_type}", flush=True)

        except KeyboardInterrupt:
            print(f"\nStopped. Last sent: {pkt}\n")


if __name__ == "__main__":
    main()
