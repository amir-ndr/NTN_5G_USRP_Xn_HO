#!/usr/bin/env python3
"""
dispatcher.py — NTN Handover Dispatcher  (Two-Level Bregman Online Learning)

Architecture
───────────────────────────────────────────────────────────────────────────────
Two-level Bregman mirror descent hierarchy (Algorithm 1, paper §III-B):

  Level 1 — PathScheduler (access layer, lives in controller.py):
      One scheduler PER TASK TYPE — each learns its own π_ISL / π_GND.
      Gradient signal: x_{i',i} = α(access_cost_i − τ_access)
      Cost function:   L(B·x) = log(1 + B·x)  where B = 1 (path-level scale)
      Update rule (positive Bregman multiplicative weights):
        log_w[chosen] += η_path · (1 / (1 + max(x, −0.999)))

      Regret_access[t] = access_cost_chosen[t] − min(cost_isl[t], cost_gnd[t])

  Level 2 — LayerScheduler (NF/compute layer, lives here):
      Learns π_AMF / π_SMF / π_UPF per layer per path.
      Gradient signal: per-instance sojourn delay (G/G/1 Kingman).
      Task-type aware via hardware-derived utilization (ρ).

      Regret_inst[t] = core_ms[t] − oracle_compute_ms[t]  (within chosen path)

  Total end-to-end latency:
      total_ms = access_cost_chosen + core_ms
             = (prop + ngap + xn_setup) + (amf + smf + upf)

  Regret decomposition:
      access_regret = access_cost_chosen − min(cost_isl, cost_gnd)
      inst_regret   = core_ms − best_compute_on_chosen_path
      global_regret = total_ms − global_oracle_ms
      global_oracle = min over {ISL, GND} of (access_cost + best_compute_on_path)

Hardware-derived instance parametrization (G/G/1 Kingman):
    capacity_hz = (millicores / 1000) × cpu_freq_ghz × 10⁹
    τ_ms        = cycles_per_req / capacity_hz × 1000
    ρ           = N_tasks × cycles_per_req / capacity_hz
    ρ ≥ 0.95    → unstable → sojourn = 999 ms

Outputs:
    dispatch_log.csv        all Bregman HOs
    isl_path_log.csv        ISL-path HOs
    ground_path_log.csv     GND-path HOs
    random_log.csv          RandomDispatcher (50/50 path + uniform NF)
"""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  Algorithm hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

N_TASKS      = 1000   # tasks per HO slot (paper §IV: 10^3 registration burst)
TAU_MAX      = 5.0    # per-instance sojourn threshold (ms) — Level-2 LayerScheduler
TAU_ACCESS   = 15.0   # access-cost threshold (ms) — Level-1 PathScheduler: x = α(c−τ_a)
ALPHA        = 0.1    # cost-signal scaling for both levels:  x = α(w − τ)
ETA_X        = 0.05   # Level-2 (NF compute) Bregman step
ETA_PATH     = 0.5    # Level-1 (path) Bregman step — positive multiplicative weights
GRAD_CAP     = 5.0    # gradient cap for both levels (prevents log-weight overflow)
PROB_FLOOR   = 0.05   # minimum probability per NF instance (exploration floor)

# ── Task types: UPF cycles per request ───────────────────────────────────────
TASK_TYPES: dict[str, int] = {
    "gaming":    783_333,    # UPF-ON-0 ρ≈0.435 (stable); GND-0 ρ≈0.112
    "youtube":   3_011_111,  # all ISL UPFs unstable (ρ>1); GND-0 ρ≈0.430
    "browsing":  2_750_000,  # all ISL UPFs unstable; GND-0 ρ≈0.393
    "instagram": 4_500_000,  # all ISL UPFs unstable; GND-0 ρ≈0.643; GND-2 unstable
    "mixed":     3_500_000,  # all ISL UPFs unstable; GND-0/1 stable
}
TASK_CYCLE = ["gaming", "youtube", "browsing", "instagram", "mixed"]

AMF_CYCLES = 400_000    # 3GPP TS 33.501 registration macro-task: 4×10^5 cycles
SMF_CYCLES = 500_000    # PDU session establishment: ~5×10^5 cycles

# ══════════════════════════════════════════════════════════════════════════════
#  Hardware instance specs
#  (name, millicores, cpu_freq_ghz, is_onboard, fixed_cpr, cs, ca)
# ══════════════════════════════════════════════════════════════════════════════

_ON_AMF_SPECS = [
    ("AMF-ON-0",  500, 1.5, True,  400_000, 0.80, 1.00),   # ρ≈0.533; τ≈0.53 ms
    ("AMF-ON-1",  350, 1.5, True,  400_000, 0.90, 1.00),   # ρ≈0.762
    ("AMF-ON-2",  200, 1.5, True,  400_000, 1.05, 1.00),   # ρ≈1.333 → unstable
]
_GND_AMF_SPECS = [
    ("AMF-GND-0", 500, 3.5, False, 400_000, 0.70, 0.80),   # ρ≈0.229; τ≈0.23 ms
    ("AMF-GND-1", 300, 3.5, False, 400_000, 0.82, 0.90),   # ρ≈0.381
    ("AMF-GND-2", 150, 3.5, False, 400_000, 0.93, 1.00),   # ρ≈0.762
]
_ON_SMF_SPECS = [
    ("SMF-ON-0",  800, 1.5, True,  500_000, 0.82, 1.00),   # ρ≈0.417; τ≈0.42 ms
    ("SMF-ON-1",  600, 1.5, True,  500_000, 0.97, 1.00),   # ρ≈0.556
    ("SMF-ON-2",  400, 1.5, True,  500_000, 1.08, 1.00),   # ρ≈0.833
]
_GND_SMF_SPECS = [
    ("SMF-GND-0", 800, 3.5, False, 500_000, 0.72, 0.82),   # ρ≈0.179; τ≈0.18 ms
    ("SMF-GND-1", 500, 3.5, False, 500_000, 0.85, 0.92),   # ρ≈0.286
    ("SMF-GND-2", 300, 3.5, False, 500_000, 0.96, 1.00),   # ρ≈0.476
]
_ON_UPF_SPECS = [
    ("UPF-ON-0",  1200, 1.5, True,  None, 0.75, 0.90),
    ("UPF-ON-1",   800, 1.5, True,  None, 0.90, 1.00),
    ("UPF-ON-2",   500, 1.5, True,  None, 1.00, 1.00),
]
_GND_UPF_SPECS = [
    ("UPF-GND-0", 2000, 3.5, False, None, 0.65, 0.75),
    ("UPF-GND-1", 1200, 3.5, False, None, 0.78, 0.88),
    ("UPF-GND-2",  800, 3.5, False, None, 0.88, 0.97),
]

# ══════════════════════════════════════════════════════════════════════════════
#  Background load functions
#  TrgSAT: fast oscillation [0.20, 0.85], period = 120 HOs
#  TN:     slow oscillation [0.10, 0.60], period = 300 HOs
# ══════════════════════════════════════════════════════════════════════════════

def trgsat_bg_load(ho_id: int) -> float:
    phase = (ho_id % 120) / 120.0
    level = 0.5 + 0.5 * math.sin(2.0 * math.pi * phase)
    return 0.20 + (0.85 - 0.20) * level


def tn_bg_load(ho_id: int) -> float:
    phase = (ho_id % 300) / 300.0
    level = 0.5 + 0.5 * math.sin(2.0 * math.pi * phase)
    return 0.10 + (0.60 - 0.10) * level


# ══════════════════════════════════════════════════════════════════════════════
#  AccessNode — Xn setup latency model
# ══════════════════════════════════════════════════════════════════════════════

class AccessNode:
    """
    Models one access node (trgSAT or TN) for Xn handover setup cost.
    xn_setup_ms = xn_base_ms / (1 − bg_load)  — stretches under load.
    total_access_cost_ms includes propagation + NGAP + Xn setup.
    """

    def __init__(self, name: str, ngap_ms: float, xn_base_ms: float):
        self.name       = name
        self.ngap_ms    = ngap_ms
        self.xn_base_ms = xn_base_ms
        self.bg_load    = 0.30

    def xn_setup_ms(self) -> float:
        return self.xn_base_ms / max(1.0 - self.bg_load, 0.05)

    def total_access_cost_ms(self, prop_ms: float) -> float:
        return prop_ms + self.ngap_ms + self.xn_setup_ms()


def compute_traffic_split(
    trgsat: AccessNode, tn: AccessNode, isl_ms: float, gnd_ms: float,
) -> tuple[float, float, float, float]:
    """Inverse-cost split — kept as reference utility; not used by PathScheduler."""
    cost_isl = trgsat.total_access_cost_ms(isl_ms)
    cost_gnd = tn.total_access_cost_ms(gnd_ms)
    w_isl    = 1.0 / max(cost_isl, 0.01)
    w_gnd    = 1.0 / max(cost_gnd, 0.01)
    total    = w_isl + w_gnd
    return w_isl / total, w_gnd / total, cost_isl, cost_gnd


# ══════════════════════════════════════════════════════════════════════════════
#  PathScheduler — Level-1 Bregman (path / access layer)
# ══════════════════════════════════════════════════════════════════════════════

class PathScheduler:
    """
    Bregman mirror descent over {ISL, GND} — one instance per task type.

    Mirrors the paper's Algorithm 1 at the access (path-selection) level:
        x_{i',i}[t] = α (access_cost_i[t] − τ_access)
        z            = max(x, −0.999)
        grad_L(x)    = 1 / (1 + z)      [gradient of log(1+z)]
        log_w[chosen] += η_path · min(grad_L, GRAD_CAP)
        log_w         -= log_w.max()    (re-centre to prevent overflow)

    Lower access cost → more negative x → larger gradient → higher weight:
    the scheduler naturally concentrates probability on the cheaper path.
    """

    def __init__(self) -> None:
        self.log_w = np.zeros(2)   # [log_w_ISL, log_w_GND]
        self.p_isl = 0.5
        self.p_gnd = 0.5

    def _compute_probs(self) -> np.ndarray:
        w = np.exp(self.log_w - self.log_w.max())
        p = w / w.sum()
        floor = PROB_FLOOR / 2          # 2.5% per path — guarantees exploration
        p = np.maximum(p, floor)
        p /= p.sum()
        return p

    def sample(self) -> str:
        """Sample path; sets p_isl/p_gnd for logging."""
        p = self._compute_probs()
        self.p_isl = float(p[0])
        self.p_gnd = float(p[1])
        return "ISL" if np.random.random() < self.p_isl else "GND"

    def update(self, chosen: str, access_cost_ms: float) -> None:
        """Update after observing the chosen path's access cost."""
        idx  = 0 if chosen == "ISL" else 1
        x    = ALPHA * (access_cost_ms - TAU_ACCESS)
        z    = max(x, -0.999)
        grad = min(1.0 / (1.0 + z), GRAD_CAP)
        self.log_w[idx] += ETA_PATH * grad
        self.log_w      -= self.log_w.max()

    def status(self) -> str:
        return f"π_ISL={self.p_isl:.3f}  π_GND={self.p_gnd:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
#  CoreInstance — hardware-parametrized G/G/1 queue
# ══════════════════════════════════════════════════════════════════════════════

class CoreInstance:
    """
    One 5G NF instance (AMF / SMF / UPF) with hardware-derived utilization.

        capacity_hz = (millicores / 1000) × cpu_freq_ghz × 10⁹
        τ_ms        = cycles_per_req / capacity_hz × 1000   (mean service time)
        ρ           = N_tasks × cycles_per_req / capacity_hz
        ρ ≥ 0.95    → unstable → sojourn = 999 ms

    G/G/1 Kingman:
        W_q = ρ/(1−ρ) · (ca²+cs²)/2 · τ
        W   = W_q + Gamma(1/cs², τ·cs²)
    """

    def __init__(
        self, name: str, millicores: int, cpu_freq_ghz: float,
        is_onboard: bool, fixed_cpr: int | None, cs: float, ca: float,
    ):
        self.name        = name
        self.is_onboard  = is_onboard
        self.fixed_cpr   = fixed_cpr
        self.cs          = cs
        self.ca          = ca
        self.capacity_hz = (millicores / 1000.0) * cpu_freq_ghz * 1e9
        self.rho    = 0.0
        self.tau_ms = 1.0
        self.B      = 1.0

    def configure(self, cpr: int, n_tasks: int) -> None:
        effective   = self.fixed_cpr if self.fixed_cpr is not None else cpr
        self.tau_ms = (effective / self.capacity_hz) * 1000.0
        self.rho    = (n_tasks * effective) / self.capacity_hz
        self.B      = 1.0 / max(self.tau_ms, 1e-9)

    def is_stable(self) -> bool:
        return self.rho < 0.95

    def _q_mean_ms(self) -> float:
        rho = min(self.rho, 0.9999)
        return (rho / (1.0 - rho)) * ((self.ca**2 + self.cs**2) / 2.0) * self.tau_ms

    def sample_delay_ms(self) -> float:
        if not self.is_stable():
            return 999.0
        k       = 1.0 / (self.cs**2)
        theta   = self.tau_ms * (self.cs**2)
        service = float(np.random.gamma(k, theta))
        return max(self._q_mean_ms() + service, 0.01)

    def expected_delay_ms(self) -> float:
        return 999.0 if not self.is_stable() else self._q_mean_ms() + self.tau_ms

    def x_signal(self, w_ms: float) -> float:
        return ALPHA * (w_ms - TAU_MAX)

    def _z(self, x: float) -> float:
        return max(self.B * x, -0.999)

    def cost(self, x: float) -> float:
        return math.log1p(self._z(x))

    def grad_L(self, x: float) -> float:
        return 1.0 / (1.0 + self._z(x))


# ══════════════════════════════════════════════════════════════════════════════
#  LayerScheduler — Level-2 Bregman (NF / compute layer)
# ══════════════════════════════════════════════════════════════════════════════

class LayerScheduler:
    """
    Bregman mirror descent on the probability simplex for one NF layer.
    Restricted to stable instances (ρ < 0.95); unstable weights decay naturally.
    Returns: (idx, probs_before, sojourn_ms, x_signal, cost, rho)
    """

    def __init__(self, instances: list[CoreInstance]):
        self.instances = instances
        self.n         = len(instances)
        self.weights   = np.ones(self.n, dtype=float)

    def probabilities(self) -> list[float]:
        return (self.weights / self.weights.sum()).tolist()

    def select_and_process(self) -> tuple[int, list[float], float, float, float, float]:
        stable = [i for i, inst in enumerate(self.instances) if inst.is_stable()]
        if not stable:
            stable = [min(range(self.n), key=lambda i: self.instances[i].rho)]

        p_all = self.weights / self.weights.sum()
        p_sel = np.zeros(self.n)
        for i in stable:
            p_sel[i] = p_all[i]
        if p_sel.sum() < 1e-12:
            for i in stable:
                p_sel[i] = 1.0
        p_sel /= p_sel.sum()

        idx          = int(np.random.choice(self.n, p=p_sel))
        probs_before = p_all.tolist()

        w_all = [inst.sample_delay_ms() if i == idx else inst.expected_delay_ms()
                 for i, inst in enumerate(self.instances)]
        x_all = [inst.x_signal(w) for inst, w in zip(self.instances, w_all)]

        for i, inst in enumerate(self.instances):
            exp = min(ETA_X * inst.grad_L(x_all[i]), GRAD_CAP)
            self.weights[i] *= math.exp(exp)
        self.weights = np.clip(self.weights, 1e-12, None)
        self.weights /= self.weights.sum()
        self.weights  = np.clip(self.weights, PROB_FLOOR / self.n, 1.0)
        self.weights /= self.weights.sum()

        chosen = self.instances[idx]
        x_val  = x_all[idx]
        return idx, probs_before, w_all[idx], x_val, chosen.cost(x_val), chosen.rho


# ══════════════════════════════════════════════════════════════════════════════
#  CSV schema
# ══════════════════════════════════════════════════════════════════════════════

_CSV_HEADER = [
    "ho_id", "timestamp", "task_type", "path",
    "isl_ms", "gnd_ms", "prop_ms",
    "path_p_isl", "path_p_gnd",
    "access_cost_isl", "access_cost_gnd",
    "trgsat_bg", "tn_bg",
    "amf_inst", "amf_p0", "amf_p1", "amf_p2",
    "amf_ms", "amf_rho", "amf_x", "amf_cost",
    "smf_inst", "smf_p0", "smf_p1", "smf_p2",
    "smf_ms", "smf_rho", "smf_x", "smf_cost",
    "upf_inst", "upf_p0", "upf_p1", "upf_p2",
    "upf_ms", "upf_rho", "upf_x", "upf_cost",
    "core_ms", "total_ms",
    "on_amf_p0", "on_amf_p1", "on_amf_p2",
    "on_smf_p0", "on_smf_p1", "on_smf_p2",
    "on_upf_p0", "on_upf_p1", "on_upf_p2",
    "gnd_amf_p0", "gnd_amf_p1", "gnd_amf_p2",
    "gnd_smf_p0", "gnd_smf_p1", "gnd_smf_p2",
    "gnd_upf_p0", "gnd_upf_p1", "gnd_upf_p2",
    "global_oracle_ms",
    "access_regret_ms", "inst_regret_ms", "global_regret_ms",
    "cum_access_regret_ms", "cum_inst_regret_ms", "cum_global_regret_ms",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Dispatcher — receives path from controller's PathScheduler, selects NFs
# ══════════════════════════════════════════════════════════════════════════════

class Dispatcher:
    """
    Level-2 Bregman: receives the path chosen by the external PathScheduler
    and selects AMF/SMF/UPF instances via Bregman mirror descent.

    total_ms = access_cost_chosen + core_ms  (full HO latency)
    """

    def __init__(self, log_dir: str | Path = "."):
        self.on_amf  = LayerScheduler([CoreInstance(*p) for p in _ON_AMF_SPECS])
        self.on_smf  = LayerScheduler([CoreInstance(*p) for p in _ON_SMF_SPECS])
        self.on_upf  = LayerScheduler([CoreInstance(*p) for p in _ON_UPF_SPECS])
        self.gnd_amf = LayerScheduler([CoreInstance(*p) for p in _GND_AMF_SPECS])
        self.gnd_smf = LayerScheduler([CoreInstance(*p) for p in _GND_SMF_SPECS])
        self.gnd_upf = LayerScheduler([CoreInstance(*p) for p in _GND_UPF_SPECS])

        self.ho_id             = 0
        self.cum_access_regret = 0.0
        self.cum_inst_regret   = 0.0
        self.cum_global_regret = 0.0

        log_dir = Path(log_dir)
        self._files:   list = []
        self._writers: dict = {}
        for key, fname in [("all", "dispatch_log.csv"),
                            ("ISL", "isl_path_log.csv"),
                            ("GND", "ground_path_log.csv")]:
            fh = open(log_dir / fname, "w", newline="")
            wr = csv.DictWriter(fh, fieldnames=_CSV_HEADER, extrasaction="ignore")
            wr.writeheader()
            self._files.append(fh)
            self._writers[key] = wr

        print(f"[Dispatcher] Two-level Bregman  "
              f"η_path={ETA_PATH}  η_x={ETA_X}  "
              f"τ_max={TAU_MAX} ms  τ_access={TAU_ACCESS} ms  N={N_TASKS} tasks")

    def _configure_all(self, task_type: str) -> None:
        upf_cpr = TASK_TYPES[task_type]
        for inst in self.on_amf.instances + self.gnd_amf.instances:
            inst.configure(AMF_CYCLES, N_TASKS)
        for inst in self.on_smf.instances + self.gnd_smf.instances:
            inst.configure(SMF_CYCLES, N_TASKS)
        for inst in self.on_upf.instances + self.gnd_upf.instances:
            inst.configure(upf_cpr, N_TASKS)

    def _best_compute_ms(self, path: str) -> float:
        """Best stable AMF+SMF+UPF compute on the given path (no access overhead)."""
        def best(sched: LayerScheduler) -> float:
            stable = [inst for inst in sched.instances if inst.is_stable()]
            return min(inst.expected_delay_ms() for inst in stable) if stable else 999.0
        if path == "ISL":
            return best(self.on_amf) + best(self.on_smf) + best(self.on_upf)
        return best(self.gnd_amf) + best(self.gnd_smf) + best(self.gnd_upf)

    def dispatch(
        self,
        path:            str,
        isl_ms:          float,
        gnd_ms:          float,
        task_type:       str   = "mixed",
        path_p_isl:      float = 0.5,
        path_p_gnd:      float = 0.5,
        access_cost_isl: float = 0.0,
        access_cost_gnd: float = 0.0,
        trgsat_bg:       float = 0.3,
        tn_bg:           float = 0.3,
    ) -> tuple[str, dict]:
        self.ho_id += 1
        self._configure_all(task_type)

        prop_ms     = isl_ms if path == "ISL" else gnd_ms
        access_cost = access_cost_isl if path == "ISL" else access_cost_gnd

        if path == "ISL":
            amf_r = self.on_amf.select_and_process()
            smf_r = self.on_smf.select_and_process()
            upf_r = self.on_upf.select_and_process()
        else:
            amf_r = self.gnd_amf.select_and_process()
            smf_r = self.gnd_smf.select_and_process()
            upf_r = self.gnd_upf.select_and_process()

        amf_idx, amf_p, amf_ms, amf_x, amf_cost, amf_rho = amf_r
        smf_idx, smf_p, smf_ms, smf_x, smf_cost, smf_rho = smf_r
        upf_idx, upf_p, upf_ms, upf_x, upf_cost, upf_rho = upf_r

        core_ms  = amf_ms + smf_ms + upf_ms
        total_ms = access_cost + core_ms   # full end-to-end HO latency

        # ── Regret decomposition ──────────────────────────────────────────────
        inst_oracle_ms   = self._best_compute_ms(path)
        inst_regret_ms   = max(0.0, core_ms - inst_oracle_ms)

        access_oracle_ms = min(access_cost_isl, access_cost_gnd)
        access_regret_ms = max(0.0, access_cost - access_oracle_ms)

        global_oracle_ms = min(
            access_cost_isl + self._best_compute_ms("ISL"),
            access_cost_gnd + self._best_compute_ms("GND"),
        )
        global_regret_ms = max(0.0, total_ms - global_oracle_ms)

        self.cum_access_regret += access_regret_ms
        self.cum_inst_regret   += inst_regret_ms
        self.cum_global_regret += global_regret_ms

        on_ap  = self.on_amf.probabilities()
        on_sp  = self.on_smf.probabilities()
        on_up  = self.on_upf.probabilities()
        gnd_ap = self.gnd_amf.probabilities()
        gnd_sp = self.gnd_smf.probabilities()
        gnd_up = self.gnd_upf.probabilities()

        row = {
            "ho_id":           self.ho_id,
            "timestamp":       datetime.now(tz=timezone.utc).isoformat(),
            "task_type":       task_type,
            "path":            path,
            "isl_ms":          round(isl_ms,  3),
            "gnd_ms":          round(gnd_ms,  3),
            "prop_ms":         round(prop_ms, 3),
            "path_p_isl":      round(path_p_isl, 4),
            "path_p_gnd":      round(path_p_gnd, 4),
            "access_cost_isl": round(access_cost_isl, 3),
            "access_cost_gnd": round(access_cost_gnd, 3),
            "trgsat_bg":       round(trgsat_bg, 4),
            "tn_bg":           round(tn_bg, 4),
            "amf_inst": amf_idx,
            "amf_p0": round(amf_p[0],4), "amf_p1": round(amf_p[1],4), "amf_p2": round(amf_p[2],4),
            "amf_ms": round(amf_ms,3), "amf_rho": round(amf_rho,4),
            "amf_x": round(amf_x,5), "amf_cost": round(amf_cost,5),
            "smf_inst": smf_idx,
            "smf_p0": round(smf_p[0],4), "smf_p1": round(smf_p[1],4), "smf_p2": round(smf_p[2],4),
            "smf_ms": round(smf_ms,3), "smf_rho": round(smf_rho,4),
            "smf_x": round(smf_x,5), "smf_cost": round(smf_cost,5),
            "upf_inst": upf_idx,
            "upf_p0": round(upf_p[0],4), "upf_p1": round(upf_p[1],4), "upf_p2": round(upf_p[2],4),
            "upf_ms": round(upf_ms,3), "upf_rho": round(upf_rho,4),
            "upf_x": round(upf_x,5), "upf_cost": round(upf_cost,5),
            "core_ms":  round(core_ms,  3),
            "total_ms": round(total_ms, 3),
            "on_amf_p0": round(on_ap[0],4), "on_amf_p1": round(on_ap[1],4), "on_amf_p2": round(on_ap[2],4),
            "on_smf_p0": round(on_sp[0],4), "on_smf_p1": round(on_sp[1],4), "on_smf_p2": round(on_sp[2],4),
            "on_upf_p0": round(on_up[0],4), "on_upf_p1": round(on_up[1],4), "on_upf_p2": round(on_up[2],4),
            "gnd_amf_p0": round(gnd_ap[0],4), "gnd_amf_p1": round(gnd_ap[1],4), "gnd_amf_p2": round(gnd_ap[2],4),
            "gnd_smf_p0": round(gnd_sp[0],4), "gnd_smf_p1": round(gnd_sp[1],4), "gnd_smf_p2": round(gnd_sp[2],4),
            "gnd_upf_p0": round(gnd_up[0],4), "gnd_upf_p1": round(gnd_up[1],4), "gnd_upf_p2": round(gnd_up[2],4),
            "global_oracle_ms":     round(global_oracle_ms,    3),
            "access_regret_ms":     round(access_regret_ms,    3),
            "inst_regret_ms":       round(inst_regret_ms,      3),
            "global_regret_ms":     round(global_regret_ms,    3),
            "cum_access_regret_ms": round(self.cum_access_regret, 3),
            "cum_inst_regret_ms":   round(self.cum_inst_regret,   3),
            "cum_global_regret_ms": round(self.cum_global_regret, 3),
        }

        self._writers["all"].writerow(row)
        self._writers[path].writerow(row)
        for fh in self._files:
            fh.flush()

        return path, row

    def status(self) -> str:
        return (f"ho_id={self.ho_id}  "
                f"cum_access={self.cum_access_regret:.1f} ms  "
                f"cum_inst={self.cum_inst_regret:.1f} ms  "
                f"cum_global={self.cum_global_regret:.1f} ms")

    def close(self) -> None:
        for fh in self._files:
            fh.close()
        print(f"[Dispatcher] Closed.  HOs={self.ho_id}  "
              f"access_regret={self.cum_access_regret:.1f} ms  "
              f"inst_regret={self.cum_inst_regret:.1f} ms  "
              f"global_regret={self.cum_global_regret:.1f} ms")


# ══════════════════════════════════════════════════════════════════════════════
#  RandomDispatcher — baseline: random path (50/50) + uniform NF selection
# ══════════════════════════════════════════════════════════════════════════════

class RandomDispatcher:
    """
    Baseline dispatcher:
      - Path: uniform random (50/50) — no learning
      - NF:   uniform random among stable instances — no weight updates

    Same schema as Dispatcher for direct comparison.
    """

    def __init__(self, log_dir: str | Path = "."):
        self.on_amf  = LayerScheduler([CoreInstance(*p) for p in _ON_AMF_SPECS])
        self.on_smf  = LayerScheduler([CoreInstance(*p) for p in _ON_SMF_SPECS])
        self.on_upf  = LayerScheduler([CoreInstance(*p) for p in _ON_UPF_SPECS])
        self.gnd_amf = LayerScheduler([CoreInstance(*p) for p in _GND_AMF_SPECS])
        self.gnd_smf = LayerScheduler([CoreInstance(*p) for p in _GND_SMF_SPECS])
        self.gnd_upf = LayerScheduler([CoreInstance(*p) for p in _GND_UPF_SPECS])

        self.ho_id             = 0
        self.cum_access_regret = 0.0
        self.cum_inst_regret   = 0.0
        self.cum_global_regret = 0.0

        log_dir  = Path(log_dir)
        self._fh = open(log_dir / "random_log.csv", "w", newline="")
        self._wr = csv.DictWriter(self._fh, fieldnames=_CSV_HEADER, extrasaction="ignore")
        self._wr.writeheader()
        print(f"[RandomDispatcher] Random path (50/50) + uniform NF  "
              f"→ {log_dir}/random_log.csv")

    def _configure_all(self, task_type: str) -> None:
        upf_cpr = TASK_TYPES[task_type]
        for inst in self.on_amf.instances + self.gnd_amf.instances:
            inst.configure(AMF_CYCLES, N_TASKS)
        for inst in self.on_smf.instances + self.gnd_smf.instances:
            inst.configure(SMF_CYCLES, N_TASKS)
        for inst in self.on_upf.instances + self.gnd_upf.instances:
            inst.configure(upf_cpr, N_TASKS)

    def _random_pick(self, sched: LayerScheduler) -> tuple[int, list[float], float, float, float, float]:
        stable = [i for i, inst in enumerate(sched.instances) if inst.is_stable()]
        if not stable:
            stable = [min(range(sched.n), key=lambda i: sched.instances[i].rho)]
        idx   = stable[int(np.random.randint(len(stable)))]
        p     = 1.0 / len(stable)
        probs = [round(p, 4) if i in stable else 0.0 for i in range(sched.n)]
        inst  = sched.instances[idx]
        w_ms  = inst.sample_delay_ms()
        x_val = inst.x_signal(w_ms)
        c_val = inst.cost(x_val)
        return idx, probs, w_ms, x_val, c_val, inst.rho

    def _best_compute_ms(self, path: str) -> float:
        def best(sched: LayerScheduler) -> float:
            stable = [inst for inst in sched.instances if inst.is_stable()]
            return min(inst.expected_delay_ms() for inst in stable) if stable else 999.0
        if path == "ISL":
            return best(self.on_amf) + best(self.on_smf) + best(self.on_upf)
        return best(self.gnd_amf) + best(self.gnd_smf) + best(self.gnd_upf)

    def dispatch(
        self,
        isl_ms:          float,
        gnd_ms:          float,
        task_type:       str   = "mixed",
        access_cost_isl: float = 0.0,
        access_cost_gnd: float = 0.0,
        trgsat_bg:       float = 0.3,
        tn_bg:           float = 0.3,
    ) -> tuple[str, dict]:
        self.ho_id += 1
        self._configure_all(task_type)

        path        = "ISL" if np.random.random() < 0.5 else "GND"
        prop_ms     = isl_ms if path == "ISL" else gnd_ms
        access_cost = access_cost_isl if path == "ISL" else access_cost_gnd

        if path == "ISL":
            amf_idx, amf_p, amf_ms, amf_x, amf_cost, amf_rho = self._random_pick(self.on_amf)
            smf_idx, smf_p, smf_ms, smf_x, smf_cost, smf_rho = self._random_pick(self.on_smf)
            upf_idx, upf_p, upf_ms, upf_x, upf_cost, upf_rho = self._random_pick(self.on_upf)
        else:
            amf_idx, amf_p, amf_ms, amf_x, amf_cost, amf_rho = self._random_pick(self.gnd_amf)
            smf_idx, smf_p, smf_ms, smf_x, smf_cost, smf_rho = self._random_pick(self.gnd_smf)
            upf_idx, upf_p, upf_ms, upf_x, upf_cost, upf_rho = self._random_pick(self.gnd_upf)

        core_ms  = amf_ms + smf_ms + upf_ms
        total_ms = access_cost + core_ms

        inst_oracle_ms   = self._best_compute_ms(path)
        inst_regret_ms   = max(0.0, core_ms - inst_oracle_ms)
        access_oracle_ms = min(access_cost_isl, access_cost_gnd)
        access_regret_ms = max(0.0, access_cost - access_oracle_ms)
        global_oracle_ms = min(
            access_cost_isl + self._best_compute_ms("ISL"),
            access_cost_gnd + self._best_compute_ms("GND"),
        )
        global_regret_ms = max(0.0, total_ms - global_oracle_ms)

        self.cum_access_regret += access_regret_ms
        self.cum_inst_regret   += inst_regret_ms
        self.cum_global_regret += global_regret_ms

        row = {
            "ho_id":           self.ho_id,
            "timestamp":       datetime.now(tz=timezone.utc).isoformat(),
            "task_type":       task_type,
            "path":            path,
            "isl_ms":          round(isl_ms,  3),
            "gnd_ms":          round(gnd_ms,  3),
            "prop_ms":         round(prop_ms, 3),
            "path_p_isl":      0.5,
            "path_p_gnd":      0.5,
            "access_cost_isl": round(access_cost_isl, 3),
            "access_cost_gnd": round(access_cost_gnd, 3),
            "trgsat_bg":       round(trgsat_bg, 4),
            "tn_bg":           round(tn_bg, 4),
            "amf_inst": amf_idx,
            "amf_p0": amf_p[0], "amf_p1": amf_p[1], "amf_p2": amf_p[2],
            "amf_ms": round(amf_ms,3), "amf_rho": round(amf_rho,4),
            "amf_x": round(amf_x,5), "amf_cost": round(amf_cost,5),
            "smf_inst": smf_idx,
            "smf_p0": smf_p[0], "smf_p1": smf_p[1], "smf_p2": smf_p[2],
            "smf_ms": round(smf_ms,3), "smf_rho": round(smf_rho,4),
            "smf_x": round(smf_x,5), "smf_cost": round(smf_cost,5),
            "upf_inst": upf_idx,
            "upf_p0": upf_p[0], "upf_p1": upf_p[1], "upf_p2": upf_p[2],
            "upf_ms": round(upf_ms,3), "upf_rho": round(upf_rho,4),
            "upf_x": round(upf_x,5), "upf_cost": round(upf_cost,5),
            "core_ms":  round(core_ms,  3),
            "total_ms": round(total_ms, 3),
            "on_amf_p0":  amf_p[0], "on_amf_p1":  amf_p[1], "on_amf_p2":  amf_p[2],
            "on_smf_p0":  smf_p[0], "on_smf_p1":  smf_p[1], "on_smf_p2":  smf_p[2],
            "on_upf_p0":  upf_p[0], "on_upf_p1":  upf_p[1], "on_upf_p2":  upf_p[2],
            "gnd_amf_p0": amf_p[0], "gnd_amf_p1": amf_p[1], "gnd_amf_p2": amf_p[2],
            "gnd_smf_p0": smf_p[0], "gnd_smf_p1": smf_p[1], "gnd_smf_p2": smf_p[2],
            "gnd_upf_p0": upf_p[0], "gnd_upf_p1": upf_p[1], "gnd_upf_p2": upf_p[2],
            "global_oracle_ms":     round(global_oracle_ms,    3),
            "access_regret_ms":     round(access_regret_ms,    3),
            "inst_regret_ms":       round(inst_regret_ms,      3),
            "global_regret_ms":     round(global_regret_ms,    3),
            "cum_access_regret_ms": round(self.cum_access_regret, 3),
            "cum_inst_regret_ms":   round(self.cum_inst_regret,   3),
            "cum_global_regret_ms": round(self.cum_global_regret, 3),
        }

        self._wr.writerow(row)
        self._fh.flush()
        return path, row

    def status(self) -> str:
        return (f"[RandomDispatcher] ho_id={self.ho_id}  "
                f"cum_global_regret={self.cum_global_regret:.1f} ms")

    def close(self) -> None:
        self._fh.close()
        print(f"[RandomDispatcher] Closed.  HOs={self.ho_id}  "
              f"global_regret={self.cum_global_regret:.1f} ms")
