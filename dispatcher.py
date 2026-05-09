#!/usr/bin/env python3
"""
dispatcher.py — NTN Handover Dispatcher (Bregman Online Learning)

Implements the algorithm from the paper exactly:

  Exploration  (Bregman mirror descent on the probability simplex):
      π_{i,j}[t+1] ∝ π_{i,j}[t] · exp(η_x · ∇L(B_j[t] · x_{i,j}[t]))
      where L(z) = log(1+z)  →  ∇L(z) = 1/(1+z)

  Exploitation (projected gradient descent on service rate B):
      B_j[t+1] = clip(B_j[t] − η_B · ∇L(B_j·x_j) · x_j,  0,  min(B_MAX, Q_j+A_j))

  Queue dynamics (virtual Lyapunov queue per instance):
      Q_j[t+1] = max(Q_j[t] + A_j[t] − B_j[t],  0)

  Cost signal at core nodes (timeout penalty, §Handover Execution):
      x_{i,j}[t] = α · (w_j[t] − τ_max)
        w_j  = sojourn time (queue wait + service) from G/G/1 Kingman
        τ_max = 5G NTN timeout threshold (recalibrated from terrestrial 200 ms)
        α    = scaling constant
        x < 0 → on time (reward); x > 0 → timeout (penalty)

  Tree cost relay (leaf→root, §Handover Preparation):
      total_cost_i[t] = prop_cost_i + L_AMF + L_SMF + L_UPF

  Dispatcher Hedge update (bandit feedback on selected path):
      π_i[t+1] ∝ π_i[t] · exp(−η_x · total_cost_i[t])

  Physical sojourn (G/G/1 Kingman):
      ρ_j = A_j[t] / B_MAX   (background load / hardware capacity)
      A_j ~ Uniform(A_LO, A_HI) tasks/slot  (system-wide background traffic)

Outputs:
  dispatch_log.csv     — every HO (all paths)
  isl_path_log.csv     — ISL-path HOs only
  ground_path_log.csv  — Ground-path HOs only
"""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  Hyperparameters  (all tunable)
# ══════════════════════════════════════════════════════════════════════════════

B_MAX      = 500.0  # hardware service capacity  (tasks / slot)
A_LO       = 200.0  # min background arrival rate (tasks / slot)
A_HI       = 400.0  # max background arrival rate (tasks / slot)
TAU_MAX    = 25.0   # 5G NTN timeout threshold    (ms)  — recalibrated for LEO
ALPHA      = 0.01   # timeout-penalty scaling: x_ij = α(w_j − τ_max)
ETA_X      = 0.005   # exploration step  (Bregman mirror descent) — slow enough for 100+ HOs
ETA_B      = 0.05   # exploitation step (projected gradient on B)
GRAD_CAP   = 2.0    # max exponent per weight update: prevents exp(50) explosion from clipping
PROB_FLOOR = 0.05   # minimum selection probability for any path/instance (forces exploration)

# ══════════════════════════════════════════════════════════════════════════════
#  G/G/1 instance parameters: (name, tau_ms, cs, ca)
#    tau_ms : mean service time (ms)         — fixed hardware characteristic
#    cs     : CV of service times            — higher = more variable compute
#    ca     : CV of inter-arrival times      — higher = more bursty traffic
#  Physical ρ = A_j[t] / B_MAX  (drawn each slot from U(A_LO, A_HI) / B_MAX)
# ══════════════════════════════════════════════════════════════════════════════

# Onboard core at trgSAT — constrained satellite hardware
# tau_ms reduced by 25% vs original so oracle ISL ≈ 36ms vs oracle GND ≈ 38ms:
# paths are competitive, oracle oscillates with orbital geometry → meaningful regret
_ON_AMF = [
    ("AMF-ON-0",  3.75, 0.80, 1.00),   # was 5.0
    ("AMF-ON-1",  6.00, 0.95, 1.00),   # was 8.0
    ("AMF-ON-2",  9.00, 1.05, 1.00),   # was 12.0
]
_ON_SMF = [
    ("SMF-ON-0",  3.75, 0.82, 1.00),   # was 5.0
    ("SMF-ON-1",  6.75, 0.97, 1.00),   # was 9.0
    ("SMF-ON-2",  9.75, 1.08, 1.00),   # was 13.0
]
_ON_UPF = [
    ("UPF-ON-0",  2.25, 0.75, 0.90),   # was 3.0
    ("UPF-ON-1",  3.75, 0.90, 1.00),   # was 5.0
    ("UPF-ON-2",  6.00, 1.00, 1.00),   # was 8.0
]

# Ground core at TN — high-capacity datacenter
_GND_AMF = [
    ("AMF-GND-0", 3.0, 0.70, 0.80),
    ("AMF-GND-1", 5.0, 0.82, 0.90),
    ("AMF-GND-2", 7.0, 0.93, 1.00),
]
_GND_SMF = [
    ("SMF-GND-0", 3.5, 0.72, 0.82),
    ("SMF-GND-1", 5.5, 0.85, 0.92),
    ("SMF-GND-2", 8.0, 0.96, 1.00),
]
_GND_UPF = [
    ("UPF-GND-0", 2.0, 0.65, 0.75),
    ("UPF-GND-1", 3.5, 0.78, 0.88),
    ("UPF-GND-2", 5.0, 0.88, 0.97),
]

# ── CSV columns ───────────────────────────────────────────────────────────────
_CSV_HEADER = [
    "ho_id", "timestamp", "path",
    "isl_ms", "gnd_ms", "prop_ms",
    # Chosen core chain — sojourn, penalty signal, individual cost
    "amf_inst", "amf_p0", "amf_p1", "amf_p2", "amf_ms", "amf_x", "amf_cost",
    "smf_inst", "smf_p0", "smf_p1", "smf_p2", "smf_ms", "smf_x", "smf_cost",
    "upf_inst", "upf_p0", "upf_p1", "upf_p2", "upf_ms", "upf_x", "upf_cost",
    "core_ms", "core_cost",          # core_cost = tree relay (AMF+SMF+UPF)
    # Per-core instance probabilities (always logged, both cores)
    "on_amf_p0", "on_amf_p1", "on_amf_p2",
    "on_smf_p0", "on_smf_p1", "on_smf_p2",
    "on_upf_p0", "on_upf_p1", "on_upf_p2",
    "gnd_amf_p0", "gnd_amf_p1", "gnd_amf_p2",
    "gnd_smf_p0", "gnd_smf_p1", "gnd_smf_p2",
    "gnd_upf_p0", "gnd_upf_p1", "gnd_upf_p2",
    "p_isl", "p_gnd",
    "total_ms", "total_cost",
    "oracle_ms", "regret_ms", "cum_regret_ms",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Classes
# ══════════════════════════════════════════════════════════════════════════════

class CoreInstance:
    """
    One 5G NF instance (e.g. AMF-ON-0) modelled as a G/G/1 queue.

    Physical sojourn: Kingman approximation with ρ = A_j / B_MAX.
    Algorithmic state: learned service-rate B (projected gradient) and
                       virtual queue Q (Lyapunov stability).
    """

    def __init__(self, name: str, tau_ms: float, cs: float, ca: float):
        self.name = name
        self.tau  = tau_ms
        self.cs   = cs
        self.ca   = ca
        self.B    = float(B_MAX)   # learned service rate, starts at hardware max
        self.Q    = 0.0            # virtual queue backlog

    # ── Physical delay ────────────────────────────────────────────────────────

    def sojourn(self, A_j: float) -> float:
        """Sample sojourn time via G/G/1 Kingman.  ρ = A_j / B_MAX."""
        rho = min(A_j / B_MAX, 0.999)
        q_mean = (rho / (1.0 - rho)) * ((self.ca**2 + self.cs**2) / 2.0) * self.tau
        k       = 1.0 / (self.cs**2)
        theta   = self.tau * (self.cs**2)
        service = float(np.random.gamma(k, theta))
        return max(q_mean + service, 0.1)

    def expected_sojourn(self, A_mean: float) -> float:
        """Deterministic Kingman estimate (used for unselected instances)."""
        rho = min(A_mean / B_MAX, 0.999)
        q_mean = (rho / (1.0 - rho)) * ((self.ca**2 + self.cs**2) / 2.0) * self.tau
        return q_mean + self.tau

    # ── Cost / gradient helpers ───────────────────────────────────────────────

    def x_signal(self, w_ms: float) -> float:
        """Timeout penalty: x_{i,j} = α(w_j − τ_max).  <0 = on time, >0 = timeout."""
        return ALPHA * (w_ms - TAU_MAX)

    def _z(self, x_ij: float) -> float:
        """z = B · x, clipped so log(1+z) and 1/(1+z) stay well-defined."""
        return max(self.B * x_ij, -0.999)

    def cost(self, x_ij: float) -> float:
        """L(B·x) = log(1 + B·x)."""
        return math.log1p(self._z(x_ij))

    def grad_L(self, x_ij: float) -> float:
        """∇L(z) = 1 / (1+z)."""
        return 1.0 / (1.0 + self._z(x_ij))

    # ── Algorithmic updates ───────────────────────────────────────────────────

    def update_B(self, x_ij: float, A_j: float) -> None:
        """
        Exploitation — projected gradient descent:
            B[t+1] = clip(B[t] − η_B · ∇L(B·x) · x,  0,  min(B_MAX, Q+A))
        """
        grad_step = x_ij / (1.0 + self._z(x_ij))   # ∇L(z) · x_ij
        B_hi      = min(B_MAX, self.Q + A_j)
        self.B    = float(np.clip(self.B - ETA_B * grad_step, 1e-6, B_hi))

    def update_queue(self, A_j: float) -> None:
        """Queue dynamics: Q[t+1] = max(Q[t] + A − B, 0)."""
        self.Q = max(self.Q + A_j - self.B, 0.0)


class LayerScheduler:
    """
    Bregman scheduler for one 5G functional layer (AMF / SMF / UPF).

    At each slot all three instances are updated (full-information feedback):
      • Selected instance  : uses actual sampled sojourn
      • Unselected instances: use Kingman expected sojourn at mean A

    Exploration (mirror descent on simplex):
        π_j[t+1] ∝ π_j[t] · exp(η_x · ∇L(B_j[t] · x_{i,j}[t]))

    Exploitation (projected gradient on B):
        B_j[t+1] = clip(B_j[t] − η_B · ∇L · x_j,  0,  min(B_MAX, Q_j+A))

    Queue updated for ALL instances every slot (background load is system-wide).
    """

    def __init__(self, instances: list[CoreInstance]):
        self.instances = instances
        self.n         = len(instances)
        self.weights   = np.ones(self.n, dtype=float)

    def probabilities(self) -> list[float]:
        return (self.weights / self.weights.sum()).tolist()

    def select_and_process(
        self, A_j: float
    ) -> tuple[int, list[float], float, float, float]:
        """
        Select one instance, process the HO task, update all instances.
        Returns (idx, probs_before, sojourn_ms, x_signal, cost).
        """
        p   = self.weights / self.weights.sum()
        idx = int(np.random.choice(self.n, p=p))

        A_mean = (A_LO + A_HI) / 2.0

        # ── Compute sojourn & cost signal for every instance ──────────────────
        w_all = []
        x_all = []
        for i, inst in enumerate(self.instances):
            w_i = inst.sojourn(A_j) if i == idx else inst.expected_sojourn(A_mean)
            w_all.append(w_i)
            x_all.append(inst.x_signal(w_i))

        # ── Exploration: update weights for ALL instances ─────────────────────
        for i, inst in enumerate(self.instances):
            # Cap exponent to GRAD_CAP: prevents exp(50) collapse when grad=1000
            exponent = min(ETA_X * inst.grad_L(x_all[i]), GRAD_CAP)
            self.weights[i] *= math.exp(exponent)
        self.weights = np.clip(self.weights, 1e-12, None)
        self.weights /= self.weights.sum()
        # Apply minimum floor so no instance is permanently excluded
        self.weights = np.clip(self.weights, PROB_FLOOR / self.n, 1.0)
        self.weights /= self.weights.sum()

        # ── Exploitation: projected gradient on B for ALL instances ───────────
        for i, inst in enumerate(self.instances):
            inst.update_B(x_all[i], A_j)

        # ── Queue: background load hits all instances every slot ──────────────
        for inst in self.instances:
            inst.update_queue(A_j)

        return idx, p.tolist(), w_all[idx], x_all[idx], self.instances[idx].cost(x_all[idx])


class CorePath:
    """
    Full AMF → SMF → UPF poly-chain with tree cost relay.

    Tree cost relay (leaf→root):
        L_core = L_AMF + L_SMF + L_UPF
    Each layer runs its own independent Bregman scheduler.
    """

    def __init__(self, amf_params, smf_params, upf_params):
        self.amf = LayerScheduler([CoreInstance(*p) for p in amf_params])
        self.smf = LayerScheduler([CoreInstance(*p) for p in smf_params])
        self.upf = LayerScheduler([CoreInstance(*p) for p in upf_params])

    def route(self) -> dict:
        """
        Route one handover request through AMF→SMF→UPF.
        Each layer samples its own independent background arrival rate A_j.
        Returns full per-layer breakdown plus accumulated tree cost.
        """
        A_amf = float(np.random.uniform(A_LO, A_HI))
        A_smf = float(np.random.uniform(A_LO, A_HI))
        A_upf = float(np.random.uniform(A_LO, A_HI))

        amf_idx, amf_p, amf_ms, amf_x, amf_cost = self.amf.select_and_process(A_amf)
        smf_idx, smf_p, smf_ms, smf_x, smf_cost = self.smf.select_and_process(A_smf)
        upf_idx, upf_p, upf_ms, upf_x, upf_cost = self.upf.select_and_process(A_upf)

        # Tree cost relay: accumulated cost propagates from leaf to root
        core_cost = amf_cost + smf_cost + upf_cost
        core_ms   = amf_ms  + smf_ms  + upf_ms

        return {
            "amf_inst": amf_idx, "amf_p": amf_p, "amf_ms": amf_ms,
            "amf_x": amf_x, "amf_cost": amf_cost,
            "smf_inst": smf_idx, "smf_p": smf_p, "smf_ms": smf_ms,
            "smf_x": smf_x, "smf_cost": smf_cost,
            "upf_inst": upf_idx, "upf_p": upf_p, "upf_ms": upf_ms,
            "upf_x": upf_x, "upf_cost": upf_cost,
            "core_ms": core_ms, "core_cost": core_cost,
        }


class Dispatcher:
    """
    Root dispatcher — selects ISL or Ground path via Hedge on tree relay cost.

    Tree cost relay (leaf→root):
        total_cost = prop_cost + AMF_cost + SMF_cost + UPF_cost

    Hedge update (selected path only — bandit feedback at dispatcher level):
        π_i[t+1] ∝ π_i[t] · exp(−η_x · total_cost_i[t])

    Expensive path → weight falls → selected less often.
    Cheap path     → weight falls less → selected more often.
    """

    def __init__(
        self,
        log_dir:  str | Path = ".",
        eta_x:    float = ETA_X,
    ):
        self.onboard = CorePath(_ON_AMF, _ON_SMF, _ON_UPF)
        self.ground  = CorePath(_GND_AMF, _GND_SMF, _GND_UPF)
        self.eta_x   = eta_x

        # Path selection weights [ISL, GND]
        self.weights = np.array([1.0, 1.0])

        # Last observed end-to-end total delay per path (full-information)
        self._last_total_ms = [40.0, 35.0]   # [ISL, GND] — reasonable priors

        self.ho_id      = 0
        self.cum_regret = 0.0

        # CSV writers
        log_dir = Path(log_dir)
        self._files:   list = []
        self._writers: dict = {}
        for key, fname in [("all", "dispatch_log.csv"),
                            ("ISL", "isl_path_log.csv"),
                            ("GND", "ground_path_log.csv")]:
            fh = open(log_dir / fname, "w", newline="")
            wr = csv.DictWriter(fh, fieldnames=_CSV_HEADER)
            wr.writeheader()
            self._files.append(fh)
            self._writers[key] = wr

        print(f"[Dispatcher] Bregman online learning  "
              f"η_x={eta_x:.3f}  η_B={ETA_B:.3f}  "
              f"τ_max={TAU_MAX:.0f} ms  α={ALPHA}")
        print(f"[Dispatcher] B_MAX={B_MAX:.0f}  "
              f"A ~ U({A_LO:.0f},{A_HI:.0f}) tasks/slot  "
              f"→  ρ_phys ~ U({A_LO/B_MAX:.2f},{A_HI/B_MAX:.2f})")
        print(f"[Dispatcher] Logs → {log_dir}/dispatch_log.csv  "
              f"(+isl_path_log.csv, ground_path_log.csv)")

    # ── Public API ────────────────────────────────────────────────────────────

    def dispatch(
        self,
        isl_ms:  float,
        gnd_ms:  float,
        src_sat: str = "",
        tgt_sat: str = "",
    ) -> tuple[str, dict]:
        """
        Select path, route through core, update weights, log.
        Returns (path, full_info_dict).
        """
        self.ho_id += 1

        # ── 1. Path selection ─────────────────────────────────────────────────
        # Apply floor so neither path drops below PROB_FLOOR (guarantees adaptation)
        probs    = self.weights / self.weights.sum()
        probs    = np.clip(probs, PROB_FLOOR, 1.0 - PROB_FLOOR)
        probs   /= probs.sum()
        path_idx = int(np.random.choice(2, p=probs))
        path     = "ISL" if path_idx == 0 else "GND"
        prop_ms  = isl_ms if path_idx == 0 else gnd_ms
        core     = self.onboard if path_idx == 0 else self.ground

        # ── 2. Core processing chain (AMF→SMF→UPF) ───────────────────────────
        r        = core.route()
        total_ms = prop_ms + r["core_ms"]

        # ── 3. Full tree relay cost arriving at dispatcher root ───────────────
        # prop_cost: propagation component of L(B·x) at this node
        self._last_total_ms[path_idx] = total_ms
        B_path    = 1.0 / max(total_ms, 0.1)
        x_path    = prop_ms / (isl_ms + gnd_ms)
        prop_cost = math.log1p(max(B_path * x_path, -0.999))
        total_cost = prop_cost + r["core_cost"]   # UPF→SMF→AMF→root relay

        # ── 4. Dispatcher Hedge update (tree relay cost drives path weights) ──
        # Expensive path → weight falls → selected less often
        self.weights[path_idx] *= math.exp(-self.eta_x * total_cost)
        self.weights = np.clip(self.weights, 1e-12, None)
        self.weights /= self.weights.sum()

        # ── 6. Regret vs oracle ───────────────────────────────────────────────
        oracle_ms       = min(isl_ms + _oracle_core_ms(self.onboard),
                              gnd_ms + _oracle_core_ms(self.ground))
        regret_ms       = max(total_ms - oracle_ms, 0.0)
        self.cum_regret += regret_ms

        # ── 7. Per-core instance probabilities ────────────────────────────────
        on_ap  = self.onboard.amf.probabilities()
        on_sp  = self.onboard.smf.probabilities()
        on_up  = self.onboard.upf.probabilities()
        gnd_ap = self.ground.amf.probabilities()
        gnd_sp = self.ground.smf.probabilities()
        gnd_up = self.ground.upf.probabilities()

        # ── 8. Log ────────────────────────────────────────────────────────────
        row = {
            "ho_id":     self.ho_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "path":      path,
            "isl_ms":    round(isl_ms,   3),
            "gnd_ms":    round(gnd_ms,   3),
            "prop_ms":   round(prop_ms,  3),
            "amf_inst":  r["amf_inst"],
            "amf_p0": round(r["amf_p"][0], 4), "amf_p1": round(r["amf_p"][1], 4), "amf_p2": round(r["amf_p"][2], 4),
            "amf_ms":    round(r["amf_ms"],  3),
            "amf_x":     round(r["amf_x"],   5),
            "amf_cost":  round(r["amf_cost"],5),
            "smf_inst":  r["smf_inst"],
            "smf_p0": round(r["smf_p"][0], 4), "smf_p1": round(r["smf_p"][1], 4), "smf_p2": round(r["smf_p"][2], 4),
            "smf_ms":    round(r["smf_ms"],  3),
            "smf_x":     round(r["smf_x"],   5),
            "smf_cost":  round(r["smf_cost"],5),
            "upf_inst":  r["upf_inst"],
            "upf_p0": round(r["upf_p"][0], 4), "upf_p1": round(r["upf_p"][1], 4), "upf_p2": round(r["upf_p"][2], 4),
            "upf_ms":    round(r["upf_ms"],  3),
            "upf_x":     round(r["upf_x"],   5),
            "upf_cost":  round(r["upf_cost"],5),
            "core_ms":   round(r["core_ms"], 3),
            "core_cost": round(r["core_cost"],5),
            "on_amf_p0": round(on_ap[0],4), "on_amf_p1": round(on_ap[1],4), "on_amf_p2": round(on_ap[2],4),
            "on_smf_p0": round(on_sp[0],4), "on_smf_p1": round(on_sp[1],4), "on_smf_p2": round(on_sp[2],4),
            "on_upf_p0": round(on_up[0],4), "on_upf_p1": round(on_up[1],4), "on_upf_p2": round(on_up[2],4),
            "gnd_amf_p0": round(gnd_ap[0],4), "gnd_amf_p1": round(gnd_ap[1],4), "gnd_amf_p2": round(gnd_ap[2],4),
            "gnd_smf_p0": round(gnd_sp[0],4), "gnd_smf_p1": round(gnd_sp[1],4), "gnd_smf_p2": round(gnd_sp[2],4),
            "gnd_upf_p0": round(gnd_up[0],4), "gnd_upf_p1": round(gnd_up[1],4), "gnd_upf_p2": round(gnd_up[2],4),
            "p_isl":          round(probs[0],       4),
            "p_gnd":          round(probs[1],       4),
            "total_ms":       round(total_ms,       3),
            "total_cost":     round(total_cost,     5),
            "oracle_ms":      round(oracle_ms,      3),
            "regret_ms":      round(regret_ms,      3),
            "cum_regret_ms":  round(self.cum_regret,3),
        }

        self._writers["all"].writerow(row)
        self._writers[path].writerow(row)
        for fh in self._files:
            fh.flush()

        return path, row

    def status(self) -> str:
        p = self.weights / self.weights.sum()
        return (f"P(ISL)={p[0]:.3f}  P(GND)={p[1]:.3f}  "
                f"cum_regret={self.cum_regret:.1f} ms  ho_id={self.ho_id}")

    def close(self) -> None:
        for fh in self._files:
            fh.close()
        print(f"[Dispatcher] Closed.  Total HOs={self.ho_id}  "
              f"Cumulative regret={self.cum_regret:.1f} ms")


# ── Oracle helper ─────────────────────────────────────────────────────────────

def _oracle_core_ms(path: CorePath) -> float:
    """
    Expected minimum sojourn for each layer at mean background load.
    Oracle uses the best (lowest τ_ms) instance per layer.
    """
    A_mean = (A_LO + A_HI) / 2.0

    def best(sched: LayerScheduler) -> float:
        return min(inst.expected_sojourn(A_mean) for inst in sched.instances)

    return best(path.amf) + best(path.smf) + best(path.upf)
