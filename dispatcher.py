#!/usr/bin/env python3
"""
dispatcher.py — NTN Handover Dispatcher (Online Learning)

At each handover the dispatcher at srcSAT selects between:
  ● ISL path  : ISL propagation → Onboard 5G core at trgSAT
                (low propagation, constrained compute → higher processing latency)
  ● Ground path: Sat-to-ground propagation → Ground 5G core at TN
                (higher propagation, abundant compute → lower processing latency)

Each core has 3 functional layers (AMF → SMF → UPF) with 3 instances each,
modelled as G/G/1 queues (Kingman's approximation).

Algorithm (from paper):
  - Path and instance selection via multiplicative weights (Hedge)
  - Service rate tracking via exponential moving average
  - Cost function: L(B·x) = log(1 + B·x), B = 1/delay, x = selection prob.
  - Weight update: w_j ← w_j · exp(η · B_j / B_ref) then renormalise
    (equivalently: exp-reward for high-throughput / low-delay instances)

Trade-off:
  Best ISL path  (ISL≈12ms + onboard≈31ms)  ≈ 43ms total
  Best GND path  (GND≈22ms + ground≈11ms)   ≈ 33ms total
  Crossover point: ISL wins when gnd_ms > ~32ms
  → Algorithm should converge to Ground path on average, switch to ISL
    when the satellite is far from TN.

Outputs:
  dispatch_log.csv     — every HO decision (all paths)
  isl_path_log.csv     — ISL-path HOs only
  ground_path_log.csv  — Ground-path HOs only
"""

from __future__ import annotations

import csv
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  G/G/1 instance parameters
#  Each tuple: (name, tau_ms, rho_lo, rho_hi, cs, ca)
#    tau_ms        = mean service time (ms)
#    rho_lo/rho_hi = utilisation range; ρ ~ Uniform(rho_lo, rho_hi) per call
#    cs            = coefficient of variation of service times
#    ca            = coefficient of variation of inter-arrival times
#
#  Kingman mean delay = [(rho/(1-rho)) * (ca²+cs²)/2 * tau] + tau
# ══════════════════════════════════════════════════════════════════════════════

# ── Onboard core at trgSAT (constrained compute) ─────────────────────────────
# Each tuple: (name, tau_ms, rho_lo, rho_hi, cs, ca)
# ρ is sampled Uniform(rho_lo, rho_hi) per call → realistic load fluctuation.
# ρ_hi capped at 0.75 so worst-case Kingman wait stays below ~60 ms;
# best instances are competitive with ground when ISL propagation is short.
_ON_AMF = [
    ("AMF-ON-0",  5.0, 0.38, 0.58, 0.80, 1.00),  # mean ρ≈0.48  d-range ≈  7–11 ms
    ("AMF-ON-1",  8.0, 0.50, 0.68, 0.95, 1.00),  # mean ρ≈0.59  d-range ≈ 16–24 ms
    ("AMF-ON-2", 12.0, 0.58, 0.75, 1.05, 1.00),  # mean ρ≈0.67  d-range ≈ 29–50 ms
]
_ON_SMF = [
    ("SMF-ON-0",  5.0, 0.40, 0.60, 0.82, 1.00),  # mean ρ≈0.50  d-range ≈  8–11 ms
    ("SMF-ON-1",  9.0, 0.52, 0.70, 0.97, 1.00),  # mean ρ≈0.61  d-range ≈ 18–29 ms
    ("SMF-ON-2", 13.0, 0.60, 0.76, 1.08, 1.00),  # mean ρ≈0.68  d-range ≈ 34–58 ms
]
_ON_UPF = [
    ("UPF-ON-0",  3.0, 0.35, 0.55, 0.75, 0.90),  # mean ρ≈0.45  d-range ≈  4–6 ms
    ("UPF-ON-1",  5.0, 0.46, 0.65, 0.90, 1.00),  # mean ρ≈0.56  d-range ≈  9–13 ms
    ("UPF-ON-2",  8.0, 0.56, 0.72, 1.00, 1.00),  # mean ρ≈0.64  d-range ≈ 18–29 ms
]

# ── Ground core at TN (high-capacity datacenter) ─────────────────────────────
# Lower ρ range → lower queuing wait, tighter variance (stable datacenter load).
_GND_AMF = [
    ("AMF-GND-0", 3.0, 0.20, 0.42, 0.70, 0.80),  # mean ρ≈0.31  d-range ≈  3–5 ms
    ("AMF-GND-1", 5.0, 0.32, 0.54, 0.82, 0.90),  # mean ρ≈0.43  d-range ≈  6–11 ms
    ("AMF-GND-2", 7.0, 0.44, 0.65, 0.93, 1.00),  # mean ρ≈0.54  d-range ≈ 10–20 ms
]
_GND_SMF = [
    ("SMF-GND-0", 3.5, 0.22, 0.44, 0.72, 0.82),  # mean ρ≈0.33  d-range ≈  4–6 ms
    ("SMF-GND-1", 5.5, 0.34, 0.56, 0.85, 0.92),  # mean ρ≈0.45  d-range ≈  7–13 ms
    ("SMF-GND-2", 8.0, 0.44, 0.66, 0.96, 1.00),  # mean ρ≈0.55  d-range ≈ 13–25 ms
]
_GND_UPF = [
    ("UPF-GND-0", 2.0, 0.18, 0.38, 0.65, 0.75),  # mean ρ≈0.28  d-range ≈  2–3.5 ms
    ("UPF-GND-1", 3.5, 0.28, 0.48, 0.78, 0.88),  # mean ρ≈0.38  d-range ≈  4–7 ms
    ("UPF-GND-2", 5.0, 0.38, 0.58, 0.88, 0.97),  # mean ρ≈0.48  d-range ≈  7–14 ms
]

# Oracle best-instance core totals (midpoint ρ, best instance per layer)
# AMF-ON-0 @ρ=0.48: 8.8 ms  SMF-ON-0 @ρ=0.50: 9.2 ms  UPF-ON-0 @ρ=0.45: 4.7 ms
_ORACLE_ONBOARD =  8.8 +  9.2 + 4.7    # ≈ 22.7 ms
_ORACLE_GROUND  =  4.0 +  5.0 + 2.8    # ≈ 11.8 ms

# ══════════════════════════════════════════════════════════════════════════════
#  Learning hyperparameters
# ══════════════════════════════════════════════════════════════════════════════
ETA_DISPATCH = 0.05   # learning rate at path-selection (dispatcher) level
ETA_LAYER    = 0.10   # learning rate at each functional layer
ETA_B        = 0.10   # EMA rate for service-rate (B) estimate
B_REF        = 1.0 / 30.0   # reference throughput (1/30ms); normalises exponents

# ══════════════════════════════════════════════════════════════════════════════

_CSV_HEADER = [
    "ho_id", "timestamp", "path",
    "isl_ms", "gnd_ms", "prop_ms",
    "amf_inst", "amf_p0", "amf_p1", "amf_p2", "amf_ms",
    "smf_inst", "smf_p0", "smf_p1", "smf_p2", "smf_ms",
    "upf_inst", "upf_p0", "upf_p1", "upf_p2", "upf_ms",
    "core_ms", "total_ms",
    # Onboard-core instance probs (always logged, regardless of chosen path)
    "on_amf_p0", "on_amf_p1", "on_amf_p2",
    "on_smf_p0", "on_smf_p1", "on_smf_p2",
    "on_upf_p0", "on_upf_p1", "on_upf_p2",
    # Ground-core instance probs (always logged)
    "gnd_amf_p0", "gnd_amf_p1", "gnd_amf_p2",
    "gnd_smf_p0", "gnd_smf_p1", "gnd_smf_p2",
    "gnd_upf_p0", "gnd_upf_p1", "gnd_upf_p2",
    "p_isl", "p_gnd",
    "oracle_ms", "regret_ms", "cum_regret_ms",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Classes
# ══════════════════════════════════════════════════════════════════════════════

class CoreInstance:
    """Single 5G core function instance modelled as a G/G/1 queue."""

    def __init__(self, name: str, tau_ms: float, rho_lo: float, rho_hi: float,
                 cs: float, ca: float):
        self.name   = name
        self.tau    = tau_ms
        self.rho_lo = rho_lo
        self.rho_hi = rho_hi
        self.cs     = cs
        self.ca     = ca
        # Initialise B using midpoint ρ (stable starting estimate)
        rho_mid = min((rho_lo + rho_hi) / 2.0, 0.999)
        self.mean_delay = (rho_mid / (1 - rho_mid)) * ((ca**2 + cs**2) / 2) * tau_ms + tau_ms
        self.B = 1.0 / self.mean_delay

    def sample_delay(self) -> float:
        """
        Sample one sojourn-time realisation:
          ρ ~ Uniform(rho_lo, rho_hi) → Kingman queuing wait (per-call) +
          service-time sample (Gamma with mean=tau, CV=cs)
        """
        rho     = min(float(np.random.uniform(self.rho_lo, self.rho_hi)), 0.999)
        q_mean  = (rho / (1 - rho)) * ((self.ca**2 + self.cs**2) / 2) * self.tau
        # Gamma(k, theta): k = 1/cs², theta = tau*cs²  →  mean=tau, CV=cs
        k       = 1.0 / (self.cs ** 2)
        theta   = self.tau * (self.cs ** 2)
        service = float(np.random.gamma(k, theta))
        return max(q_mean + service, 0.1)

    def update_B(self, realised_delay: float) -> None:
        """EMA update: B converges toward actual throughput 1/d."""
        self.B = ETA_B * (1.0 / max(realised_delay, 0.1)) + (1 - ETA_B) * self.B


class LayerScheduler:
    """
    Hedge scheduler for one 5G functional layer (3 instances).
    Maintains a probability vector over instances; updates after every task.
    """

    def __init__(self, instances: list[CoreInstance], eta: float = ETA_LAYER):
        self.instances = instances
        self.n         = len(instances)
        self.eta       = eta
        self.weights   = np.ones(self.n)       # unnormalised Hedge weights

    def probabilities(self) -> list[float]:
        p = self.weights / self.weights.sum()
        return p.tolist()

    def select_and_process(self) -> tuple[int, list[float], float]:
        """
        Sample an instance (proportional to weights), process one task,
        update weights.  Returns (instance_idx, probs_before_update, delay_ms).
        """
        p   = self.weights / self.weights.sum()
        idx = int(np.random.choice(self.n, p=p))
        inst = self.instances[idx]

        delay = inst.sample_delay()

        # Multiplicative weight update: reward ∝ throughput
        # w_j ← w_j · exp(η · B_j / B_ref)  (only for the selected instance)
        B = 1.0 / max(delay, 0.1)
        self.weights[idx] *= math.exp(self.eta * B / B_REF)
        self.weights = np.clip(self.weights, 1e-10, None)
        self.weights /= self.weights.sum()      # renormalise for stability

        inst.update_B(delay)
        return idx, p.tolist(), delay


class CorePath:
    """Full AMF → SMF → UPF processing chain for one path."""

    def __init__(
        self,
        amf_params: list,
        smf_params: list,
        upf_params: list,
        eta: float = ETA_LAYER,
    ):
        self.amf = LayerScheduler([CoreInstance(*p) for p in amf_params], eta=eta)
        self.smf = LayerScheduler([CoreInstance(*p) for p in smf_params], eta=eta)
        self.upf = LayerScheduler([CoreInstance(*p) for p in upf_params], eta=eta)

    def route(self) -> dict:
        """
        Route one task through AMF → SMF → UPF.
        Returns a dict with per-layer indices, probabilities, delays, and total.
        """
        amf_idx, amf_p, amf_ms = self.amf.select_and_process()
        smf_idx, smf_p, smf_ms = self.smf.select_and_process()
        upf_idx, upf_p, upf_ms = self.upf.select_and_process()
        return {
            "amf_inst": amf_idx, "amf_p": amf_p, "amf_ms": amf_ms,
            "smf_inst": smf_idx, "smf_p": smf_p, "smf_ms": smf_ms,
            "upf_inst": upf_idx, "upf_p": upf_p, "upf_ms": upf_ms,
            "core_ms":  amf_ms + smf_ms + upf_ms,
        }


class Dispatcher:
    """
    Top-level handover dispatcher.

    At each HO it:
      1. Selects ISL or Ground path via Hedge weights
      2. Routes the HO task through the chosen core (AMF→SMF→UPF)
      3. Updates path weights from the realised total delay
      4. Logs the full breakdown to CSV

    Usage
    -----
    Create once at startup:
        d = Dispatcher(log_dir="/path/to/project")

    Call at each HO:
        path, info = d.dispatch(isl_ms=12.3, gnd_ms=22.5)
        new_dest = "trgSAT" if path == "ISL" else "TN"

    Close on exit:
        d.close()
    """

    def __init__(
        self,
        log_dir:      str | Path = ".",
        eta_dispatch: float = ETA_DISPATCH,
        eta_layer:    float = ETA_LAYER,
    ):
        self.onboard = CorePath(_ON_AMF,  _ON_SMF,  _ON_UPF,  eta=eta_layer)
        self.ground  = CorePath(_GND_AMF, _GND_SMF, _GND_UPF, eta=eta_layer)
        self.eta     = eta_dispatch

        # Dispatcher-level Hedge weights [ISL, GND]
        self.weights = np.array([0.5, 0.5])

        self.ho_id      = 0
        self.cum_regret = 0.0

        # ── CSV writers ───────────────────────────────────────────────────────
        log_dir = Path(log_dir)
        self._files   = []
        self._writers = {}

        for key, fname in [("all", "dispatch_log.csv"),
                            ("ISL", "isl_path_log.csv"),
                            ("GND", "ground_path_log.csv")]:
            fh = open(log_dir / fname, "w", newline="")
            wr = csv.DictWriter(fh, fieldnames=_CSV_HEADER)
            wr.writeheader()
            self._files.append(fh)
            self._writers[key] = wr

        print(f"[Dispatcher] Onboard core best≈{_ORACLE_ONBOARD:.1f} ms  "
              f"Ground core best≈{_ORACLE_GROUND:.1f} ms")
        print(f"[Dispatcher] Logs → {log_dir}/dispatch_log.csv  "
              f"(+isl_path_log.csv, ground_path_log.csv)")

    # ── Public API ────────────────────────────────────────────────────────────

    def dispatch(
        self,
        isl_ms: float,
        gnd_ms: float,
        src_sat: str = "",
        tgt_sat: str = "",
    ) -> tuple[str, dict]:
        """
        Select a path, route the HO task, update weights, log result.

        Returns
        -------
        path : "ISL" or "GND"
        info : dict with full latency breakdown (suitable for print)
        """
        self.ho_id += 1

        # ── 1. Path selection ─────────────────────────────────────────────────
        probs     = self.weights / self.weights.sum()
        path_idx  = int(np.random.choice(2, p=probs))
        path      = "ISL" if path_idx == 0 else "GND"
        prop_ms   = isl_ms if path_idx == 0 else gnd_ms
        core      = self.onboard if path_idx == 0 else self.ground

        # ── 2. Core processing ────────────────────────────────────────────────
        r         = core.route()
        total_ms  = prop_ms + r["core_ms"]

        # ── 3. Dispatcher Hedge update ─────────────────────────────────────────
        B_total = 1.0 / max(total_ms, 0.1)
        self.weights[path_idx] *= math.exp(self.eta * B_total / B_REF)
        self.weights = np.clip(self.weights, 1e-10, None)
        self.weights /= self.weights.sum()

        # ── 4. Regret (oracle uses best fixed instances) ───────────────────────
        oracle_ms = min(
            isl_ms + _ORACLE_ONBOARD,
            gnd_ms + _ORACLE_GROUND,
        )
        regret_ms       = total_ms - oracle_ms
        self.cum_regret += regret_ms

        # ── 5. Per-core instance probs (both cores, post-update) ──────────────
        on_ap  = self.onboard.amf.probabilities()
        on_sp  = self.onboard.smf.probabilities()
        on_up  = self.onboard.upf.probabilities()
        gnd_ap = self.ground.amf.probabilities()
        gnd_sp = self.ground.smf.probabilities()
        gnd_up = self.ground.upf.probabilities()

        # ── 6. Logging ────────────────────────────────────────────────────────
        row = {
            "ho_id":        self.ho_id,
            "timestamp":    datetime.now(tz=timezone.utc).isoformat(),
            "path":         path,
            "isl_ms":       round(isl_ms, 3),
            "gnd_ms":       round(gnd_ms, 3),
            "prop_ms":      round(prop_ms, 3),
            "amf_inst":     r["amf_inst"],
            "amf_p0":       round(r["amf_p"][0], 4),
            "amf_p1":       round(r["amf_p"][1], 4),
            "amf_p2":       round(r["amf_p"][2], 4),
            "amf_ms":       round(r["amf_ms"],  3),
            "smf_inst":     r["smf_inst"],
            "smf_p0":       round(r["smf_p"][0], 4),
            "smf_p1":       round(r["smf_p"][1], 4),
            "smf_p2":       round(r["smf_p"][2], 4),
            "smf_ms":       round(r["smf_ms"],  3),
            "upf_inst":     r["upf_inst"],
            "upf_p0":       round(r["upf_p"][0], 4),
            "upf_p1":       round(r["upf_p"][1], 4),
            "upf_p2":       round(r["upf_p"][2], 4),
            "upf_ms":       round(r["upf_ms"],  3),
            "core_ms":      round(r["core_ms"], 3),
            "total_ms":     round(total_ms,     3),
            "on_amf_p0":    round(on_ap[0],  4), "on_amf_p1":  round(on_ap[1],  4), "on_amf_p2":  round(on_ap[2],  4),
            "on_smf_p0":    round(on_sp[0],  4), "on_smf_p1":  round(on_sp[1],  4), "on_smf_p2":  round(on_sp[2],  4),
            "on_upf_p0":    round(on_up[0],  4), "on_upf_p1":  round(on_up[1],  4), "on_upf_p2":  round(on_up[2],  4),
            "gnd_amf_p0":   round(gnd_ap[0], 4), "gnd_amf_p1": round(gnd_ap[1], 4), "gnd_amf_p2": round(gnd_ap[2], 4),
            "gnd_smf_p0":   round(gnd_sp[0], 4), "gnd_smf_p1": round(gnd_sp[1], 4), "gnd_smf_p2": round(gnd_sp[2], 4),
            "gnd_upf_p0":   round(gnd_up[0], 4), "gnd_upf_p1": round(gnd_up[1], 4), "gnd_upf_p2": round(gnd_up[2], 4),
            "p_isl":        round(probs[0], 4),
            "p_gnd":        round(probs[1], 4),
            "oracle_ms":    round(oracle_ms,    3),
            "regret_ms":    round(regret_ms,    3),
            "cum_regret_ms": round(self.cum_regret, 3),
        }

        self._writers["all"].writerow(row)
        self._writers[path].writerow(row)
        for fh in self._files:
            fh.flush()

        return path, row

    def status(self) -> str:
        """One-line status string for printing."""
        p = self.weights / self.weights.sum()
        return (f"P(ISL)={p[0]:.3f}  P(GND)={p[1]:.3f}  "
                f"cum_regret={self.cum_regret:.1f}ms  ho_id={self.ho_id}")

    def close(self) -> None:
        for fh in self._files:
            fh.close()
        print(f"[Dispatcher] Closed logs. Total HOs dispatched: {self.ho_id}  "
              f"Cumulative regret: {self.cum_regret:.1f} ms")
