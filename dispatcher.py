#!/usr/bin/env python3
"""
dispatcher.py — NTN Handover Dispatcher  (Two-Level Bregman Online Learning)

Architecture
───────────────────────────────────────────────────────────────────────────────
Two-level Bregman mirror descent hierarchy (Algorithm 1, paper §III-B):

  Level 1 — PathScheduler (access layer, lives in controller.py):
      Single shared scheduler across all task types — accumulates all 175 HO updates
      into one π_ISL / π_GND estimate (vs. ~35 updates per scheduler with per-task dict).
      Gradient signal: x_i = prop_delay_i  (ms, propagation cost of path i)
      Cost function:   L(x) = log(1 + B_i · x)  with B_i = 1/xn_setup_i (inverse Xn delay)
      Update rule (inverse-cost oracle via Bregman mirror descent):
        B_i[t] ← 1 / xn_setup_i  (set directly, no gradient update needed)
        log_w[i] += η_path · min(B_i/(1 + B_i·x_i), GRAD_CAP)
      With this choice, the exploration gradient becomes:
        B_i/(1+B_i·x_i) = (1/d_xn)/(1 + x_prop/d_xn) = 1/(d_xn + x_prop) = 1/access_cost
      So π converges to inverse-cost optimal split automatically.
      After each HO, B[0]/B[1] are pushed into trgsat_node.B/tn_node.B so that
      AccessNode.xn_setup_ms() reflects the algorithm's actual dispatch rate (closed loop).

      Regret_access[t] = access_cost_chosen[t] − min(cost_isl[t], cost_gnd[t])

  Level 2 — LayerScheduler (NF/compute layer, lives here):
      Learns π_AMF / π_SMF / π_UPF per layer per path.
      Gradient signal: per-instance sojourn delay (G/G/1 Kingman).
      Update rule:  weights[i] *= exp(η_x · min(B_i/(1+B_i·x_i), GRAD_CAP))
        where x_i = α(w_i − τ_max),  B_i = 1/τ_i  (service rate of instance i).
      UPF has one LayerScheduler PER TASK TYPE (per-task CPR → different ρ → different
      learned weights). AMF/SMF are shared across task types (fixed_cpr is task-independent).

      Local-only gradient (no recursive cost relay): AMF, SMF, and UPF are independent
      queues — AMF's instance choice does not affect SMF or UPF load, so propagating
      downstream costs into upstream gradient signals would penalise correct AMF choices
      whenever UPF is overloaded (wrong signal attribution).

      B-parameter update (exploitation arm): B_i starts at 1/τ_i (hardware service rate)
      and is updated each HO via exploit_update() using a projected gradient step:
        B_new = B_i − η_B · (x/(1+B·x)),   projected onto [B_MULT_MIN, B_MULT_MAX] · B_hw
      The gradient ∂L/∂B = x/(1+B·x) is dimensionless; product with B_hw yields 1/ms units.
      B_mult persists across task-type switches; base_rho at B_mult=1.0 is stored as a
      fixed oracle reference so the oracle does not drift as the algorithm learns.

      Queue state across HOs: G/G/1 steady-state captures queuing within a round.
      HO events are ~5400 simulated seconds apart; queues drain between events, so
      per-round stateless queuing is appropriate at this time scale.

      Regret_inst[t] = core_ms[t] − oracle_compute_ms[t]  (within chosen path)
      oracle_compute_ms = optimal equal-split across stable instances (Jensen lower bound)

  Total end-to-end latency:
      total_ms = access_cost_chosen + core_ms
             = (prop + ngap + xn_setup) + (amf + smf + upf)

  Regret decomposition:
      access_regret = access_cost_chosen − min(cost_isl, cost_gnd)
      inst_regret   = core_ms − best_compute_on_chosen_path
      global_regret = total_ms − global_oracle_ms
      global_oracle = min over {ISL, GND} of (access_cost + best_compute_on_path)

Hardware-derived instance parametrization (G/G/1 Kingman):
    capacity_hz  = (millicores / 1000) × cpu_freq_ghz × 10⁹
    τ_ms         = cycles_per_req / capacity_hz × 1000
    base_rho     = N_tasks × cycles_per_req / capacity_hz         (oracle: full load, x_ij=1.0)
    effective_rho = N_tasks × x_ij × cycles_per_req / capacity_hz (actual routed load)
    ρ            = B_mult × effective_rho
    ρ ≥ 0.95     → unstable (at current routing fraction) → sojourn = 999 ms
    base_rho ≥ 0.95 → physically saturated → exploit_update() skipped

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

# Per-task concurrent request counts — affects ρ = N × CPR / capacity_hz.
# Differentiated by traffic profile: gaming is a single session with low packet
# rate; instagram involves many simultaneous media requests at high volume.
# N also drives AMF/SMF utilisation (fixed CPR) so all layers are task-aware.
TASK_N_TASKS: dict[str, int] = {
    "gaming":    400,   # single gaming session, low packet rate
    "youtube":   700,   # streaming + metadata, moderate burst
    "browsing":  900,   # multiple tabs, DNS + page requests
    "instagram": 1000,  # media-heavy, many small concurrent requests
    "mixed":     800,   # average of above
}

# Global load scale factor — multiply all N_r values by this at runtime.
# Set via controller.py --load-scale or directly before Dispatcher() is constructed.
#   0.5 = light load   (all NTN-UPF stable under instagram)
#   1.0 = default load (NTN-UPF-ON0/1 unstable under instagram)
#   1.5 = heavy load   (all NTN-UPF unstable under instagram, GND takes over)
#   2.0 = extreme load (NTN-AMF-2 also unstable)
LOAD_SCALE: float = 1.0

# TAU_ACCESS is no longer used by PathScheduler (paper-correct implementation uses
# Xn G/G/1 service rate directly).  Kept here only for any legacy references.
TAU_ACCESS   = 42.0

# Per-instance τ_max is embedded in each CoreInstance spec tuple (8th element).
# Rationale: TN (GND) instances run on high-clock server CPUs → stricter deadlines;
# NTN (ON) instances run on power-limited onboard CPUs → relaxed deadlines.
# Values chosen so each instance's expected delay sits near its τ_max at mid-load,
# giving the LayerScheduler gradient differentiation across load regimes.
ALPHA        = 0.1    # cost-signal scaling for Level-2:  x = α(w − τ_max)

# Step sizes — calibrated for T = HO_HARD_CAP via η = c/√T theory.
# ETA_PATH drives path-level Bregman updates (Level-1).
# ETA_X    drives NF instance weight updates (Level-2 exploration).
# ETA_B    drives per-instance B_mult updates (Level-2 exploitation).
_T           = 300                        # nominal horizon = HO_HARD_CAP
ETA_PATH     = 0.05                      # reduced for Fix 1b: new gradient x/(1+Bx) is ~7x larger than old
ETA_X        = 0.1 #1.0 / math.sqrt(_T)       # ≈ 0.050
ETA_B        = 0.01 #0.5  / math.sqrt(_T)     # reduced from 0.1 for Fix 1a

GRAD_CAP     = 5.0    # gradient cap for both levels (prevents log-weight overflow)
PROB_FLOOR   = 0.08   # minimum probability per NF instance (exploration floor)
B_MULT_MIN   = 0.7    # minimum B_mult (70% of hardware baseline) — was 0.5
B_MULT_MAX   = 1.5    # maximum B_mult (150% of hardware baseline) — was 2.0

# Level-1 PathScheduler B parameters.
# B_i is a SERVICE RATE (1/ms) — NOT a delay.
# B_i · x_{·,i} = (1/ms) × ms = dimensionless, keeping grad = B/(1+B·x) finite.
#
# Tuned so the paper-correct dynamic projection bound min{B_max, Q + γ} actually
# binds during operation (rather than the static B_PATH_MAX). For chosen path,
# γ = 1/τ ≈ 0.20–0.25; B_PATH_MAX = 0.20 keeps the dynamic bound dominant when Q
# is small. RHO_MIN_PATH = 0.02 lets idle paths' B genuinely collapse, exposing
# the load-coupling feedback the paper specifies.
B_PATH_INIT  = 0.05   # start below initial dynamic cap so B has room to evolve
B_PATH_MAX   = 0.40   # must exceed γ_max = 1/τ_min = 1/4.0 = 0.25 so Q can drain
RHO_MAX_PATH = 0.80   # ρ cap enforced per-node in the projection step (unused now)
RHO_MIN_PATH = 0.02   # permissive floor — let idle path's B drop near zero

# ── Task types: UPF CPR fallback (used only if instance has no task_cpr dict) ──
# With per-instance task_cpr, these values are superseded for UPF instances.
# They remain authoritative for any future UPF instance added without a task_cpr dict.
TASK_TYPES: dict[str, int] = {
    "gaming":    783_333,
    "youtube":   1_400_000,
    "browsing":  1_100_000,
    "instagram": 1_600_000,
    "mixed":     1_200_000,
}
TASK_CYCLE = ["gaming", "youtube", "browsing", "instagram", "mixed"]

AMF_CYCLES = 400_000    # 3GPP TS 33.501 registration macro-task: 4×10^5 cycles
SMF_CYCLES = 500_000    # PDU session establishment: ~5×10^5 cycles

# ══════════════════════════════════════════════════════════════════════════════
#  Hardware instance specs
#  (name, millicores, cpu_freq_ghz, is_onboard, fixed_cpr, cs, ca, tau_max_ms)
#
#  τ_max_ms is per-instance, per-stage, per-path (paper §III-B "timeout thresholds
#  across each functional stage").  Design rule:
#    ON  instances: relaxed — onboard CPUs are slower, higher variance acceptable
#    GND instances: strict  — server CPUs are fast; delay above threshold is a signal
#  Each value is set near the mid-load expected sojourn so the gradient is informative
#  across the full load range (not clamped to the boundary on either extreme).
# ══════════════════════════════════════════════════════════════════════════════

#  Capacity design rule: every instance must be STABLE (ρ < 0.95) at B_mult=1.0
#  for every task, so the LayerScheduler always has 3 live choices to start from.
#  Ranking differentiation comes from: (1) raw τ (faster instance = lower base delay),
#  (2) B_mult evolution — the exploitation arm raises B_mult for underloaded instances
#      and lowers it for overloaded ones, creating non-stationary ρ crossings over time,
#  (3) variability cs/ca (Kingman variance term differentiates instances).
#  ON cs/ca higher than GND → satellite NF stack more bursty than data-centre.

_ON_AMF_SPECS = [
    # cap(MHz): ON-0=1050  ON-1=750  ON-2=600  →  base ρ(inst=1000, cpr=400k) = 0.38/0.53/0.67
    # All stable at B_mult=1.0; exploitation arm creates load crossings over time.
    ("AMF-ON-0",  700, 1.5, True,  400_000, 1.10, 1.20, 1.5),
    ("AMF-ON-1",  500, 1.5, True,  400_000, 1.25, 1.20, 1.5),
    ("AMF-ON-2",  400, 1.5, True,  400_000, 1.40, 1.20, 1.5),
]
_GND_AMF_SPECS = [
    # cap(MHz): GND-0=1750  GND-1=1225  GND-2=875  → base ρ = 0.23/0.33/0.46
    # All stable at B_mult=1.0; low cs/ca keeps Kingman W_q bounded.
    ("AMF-GND-0", 500, 3.5, False, 400_000, 0.55, 0.70, 0.5),
    ("AMF-GND-1", 350, 3.5, False, 400_000, 0.65, 0.75, 0.5),
    ("AMF-GND-2", 250, 3.5, False, 400_000, 0.75, 0.80, 0.5),
]
_ON_SMF_SPECS = [
    # cap(MHz): ON-0=1350  ON-1=1050  ON-2=825  → base ρ(inst=1000, cpr=500k) = 0.37/0.48/0.61
    # All stable at B_mult=1.0; exploitation arm creates load crossings over time.
    ("SMF-ON-0",  900, 1.5, True,  500_000, 1.05, 1.15, 1.0),
    ("SMF-ON-1",  700, 1.5, True,  500_000, 1.20, 1.15, 1.0),
    ("SMF-ON-2",  550, 1.5, True,  500_000, 1.35, 1.15, 1.0),
]
_GND_SMF_SPECS = [
    # cap(MHz): GND-0=2800  GND-1=1750  GND-2=1225  → base ρ = 0.18/0.29/0.41
    # All stable at B_mult=1.0; low cs/ca keeps delay variance small.
    ("SMF-GND-0", 800, 3.5, False, 500_000, 0.55, 0.65, 0.3),
    ("SMF-GND-1", 500, 3.5, False, 500_000, 0.65, 0.75, 0.3),
    ("SMF-GND-2", 350, 3.5, False, 500_000, 0.75, 0.80, 0.3),
]
# ── Per-instance UPF CPR matrix ───────────────────────────────────────────────
# Each UPF instance has a different software implementation optimised for a
# different traffic pattern.  CPR (cycles per request) varies per instance per
# task type, causing relative ρ rankings to genuinely flip across task switches.
#
#   ON-0  (DPDK user-space, 1800 MHz): cheap small-packet (gaming/browsing) but
#          expensive per-byte (instagram → ρ > 1 → UNSTABLE).
#   ON-1  (kernel-space, 1350 MHz):    balanced; mid-CPR everywhere; unstable for
#          instagram only.
#   ON-2  (batch processor, 1350 MHz): expensive for tiny gaming packets but
#          efficient for media — STAYS STABLE under instagram (the only ON UPF
#          that does). LayerScheduler must learn to pick it specifically when
#          task=instagram and path=ISL.
#
# ρ per task (cap: ON-0=1.8 GHz, ON-1=1.35 GHz, ON-2=1.35 GHz):
#   gaming(N=400):    ON-0 ρ≈0.11  ON-1 ρ≈0.24  ON-2 ρ≈0.44   ON-0 best
#   youtube(N=700):   ON-0 ρ≈0.39  ON-1 ρ≈0.57  ON-2 ρ≈0.42   ON-0 best
#   browsing(N=900):  ON-0 ρ≈0.50  ON-1 ρ≈0.67  ON-2 ρ≈0.57   ON-0 best
#   instagram(N=1000):ON-0 ρ≈1.11  ON-1 ρ≈1.19  ON-2 ρ≈0.67   ON-2 ONLY STABLE
#   mixed(N=800):     ON-0 ρ≈0.53  ON-1 ρ≈0.65  ON-2 ρ≈0.53   ON-0 ≈ ON-2 (tie)
_UPF_CPR_ON0  = {"gaming":  500_000,  "youtube": 1_000_000, "browsing": 1_000_000,
                  "instagram": 2_000_000, "mixed": 1_200_000}
_UPF_CPR_ON1  = {"gaming":  800_000,  "youtube": 1_100_000, "browsing": 1_000_000,
                  "instagram": 1_600_000, "mixed": 1_100_000}
_UPF_CPR_ON2  = {"gaming": 1_500_000, "youtube":   800_000, "browsing":   850_000,
                  "instagram":  900_000, "mixed":     900_000}

# GND UPF caps: GND-0=7.0 GHz, GND-1=4.2 GHz, GND-2=2.45 GHz. All stable everywhere
# (max ρ ≤ 0.40), so they form a reliable fallback. GND-0 always fastest, but its
# raw cost premium (server-class hw) is paid in higher access cost via PathScheduler.
_UPF_CPR_GND0 = {"gaming":  400_000,  "youtube": 1_000_000, "browsing":   800_000,
                  "instagram": 1_300_000, "mixed":   850_000}
_UPF_CPR_GND1 = {"gaming":  600_000,  "youtube":   900_000, "browsing":   850_000,
                  "instagram": 1_100_000, "mixed":   900_000}
_UPF_CPR_GND2 = {"gaming":  900_000,  "youtube":   700_000, "browsing":   750_000,
                  "instagram":   800_000, "mixed":   750_000}

# cs/ca: ON instances have HIGHER variability (bursty satellite NF) than GND.
_ON_UPF_SPECS = [
    ("UPF-ON-0",  1200, 1.5, True,  None, 1.05, 1.15, 2.0, _UPF_CPR_ON0),
    ("UPF-ON-1",   900, 1.5, True,  None, 1.20, 1.15, 2.0, _UPF_CPR_ON1),
    ("UPF-ON-2",   900, 1.5, True,  None, 1.35, 1.15, 2.0, _UPF_CPR_ON2),
]
_GND_UPF_SPECS = [
    ("UPF-GND-0", 2000, 3.5, False, None, 0.55, 0.65, 0.4, _UPF_CPR_GND0),
    ("UPF-GND-1", 1200, 3.5, False, None, 0.65, 0.75, 0.4, _UPF_CPR_GND1),
    ("UPF-GND-2",  700, 3.5, False, None, 0.75, 0.80, 0.4, _UPF_CPR_GND2),
]

# ══════════════════════════════════════════════════════════════════════════════
#  AccessNode — Xn setup latency model
# ══════════════════════════════════════════════════════════════════════════════

class AccessNode:
    """
    Models one access node (trgSAT or TN) for Xn handover setup cost.

    total_access_cost = prop_ms + xn_setup_ms
      prop_ms     — orbital propagation delay (real Skyfield geometry).
                    Typical: ISL ≈ 15–22 ms, GND ≈ 25–32 ms.
      xn_setup_ms — full G/G/1 Kingman sojourn (paper Eq. 3 / Algorithm 1 d_i[t]):
                      W_q = ρ/(1−ρ) × (c_a² + c_s²)/2 × τ
                      d_i = W_q + τ
                    This is the same formula used for NF instances and is the d_i[t]
                    fed into the B_i exploitation update (Algorithm 1, Eq. 11).

    PHASE TRANSITION DESIGN (Level-1 PathScheduler tradeoff)
    ────────────────────────────────────────────────────────
    ρ for each access node is driven by the PathScheduler's exploitation estimate B:
        ρ = B × xn_base_ms   (B in 1/ms, τ = xn_base_ms in ms → dimensionless)
    PathScheduler starts both paths at B_PATH_INIT = 0.125, then updates B via its
    exploitation arm each HO. After the update, it pushes B[0] → trgsat_node.B and
    B[1] → tn_node.B, closing the feedback loop: heavier ISL use → higher B_ISL →
    higher ρ_ISL → xn inflates → gradient weakens → π_ISL falls → equilibrium.

    Physics of xn_base_ms (baseline τ = 1/μ — mean service time when queue is empty):
      TrgSAT (ISL, 4.0 ms): onboard satellite Xn processor — LOW capacity, bursty
                            signaling (high c_s, high c_a).  G/G/1 Kingman inflates
                            quickly with ρ AND is amplified by the variance terms,
                            correctly reflecting satellite Xn unpredictability.
      TN     (GND, 5.0 ms): ground core Xn signaling traverses more hops
                            (gNB → AMF → anchor UPF) — HIGHER baseline, but ground
                            servers have stable service (low c_s, low c_a), so the
                            variance multiplier is smaller and d_i stays bounded.

    Coefficients of variation (paper G/G/1 notation):
      c_a — CoV of inter-arrival times (burstiness of incoming Xn requests)
      c_s — CoV of service times (variability of Xn processing duration)
      For M/M/1: c_a = c_s = 1.0  (Poisson arrivals, exponential service)
      TrgSAT uses c_s=1.2, c_a=1.1 — slightly super-Poisson (bursty onboard CPU)
      TN     uses c_s=0.6, c_a=0.7 — sub-Poisson (deterministic ground processing)
    """

    def __init__(self, name: str, ngap_ms: float, xn_base_ms: float,
                 xn_capacity_hz: float = 4e9,
                 cs: float = 1.0, ca: float = 1.0):
        self.name           = name
        self.ngap_ms        = ngap_ms      # kept for reference; excluded from cost signal
        self.xn_base_ms     = xn_base_ms
        self.xn_capacity_hz = xn_capacity_hz
        self.cs             = cs           # CoV of service time  (paper G/G/1 d_i)
        self.ca             = ca           # CoV of inter-arrival time
        self.B              = B_PATH_INIT  # dispatch rate (1/ms); updated by PathScheduler

        # ── Xn dispatcher queue state (paper Algorithm 1, Eq. 1 at i=1) ───────
        # Q[t+1] = max(Q[t] + γ[t] − B[t], 0)  evolves once per HO round.
        # Used by PathScheduler exclusively as the dynamic projection bound on B
        # (paper Step 1b: B[t+1] ∈ [0, min{B_max, Q[t] + γ[t]}]).  NOT used in
        # delay computation — d_i is still G/G/1 steady-state via xn_setup_ms().
        # Units: Q and γ are in 1/ms (service-rate-equivalent), matching B.
        # One HO arrival adds γ = 1/τ units (one mean service time of work).
        self.Q     = 0.0
        self.gamma = 0.0

    def update_queue(self, chosen: bool) -> None:
        """Evolve dispatcher backlog Q per paper Eq. 1.  Called by PathScheduler
        AFTER B has been re-projected this round (paper order: project B[t+1],
        then evolve Q[t+1]).  γ = 1/τ if this path was selected, else 0."""
        self.gamma = (1.0 / max(self.xn_base_ms, 1e-9)) if chosen else 0.0
        self.Q     = max(self.Q + self.gamma - self.B, 0.0)

    def xn_setup_ms(self) -> float:
        # Full G/G/1 Kingman formula — matches paper Eq. 3 / Algorithm 1 d_i[t].
        # ρ = B × τ where B is the PathScheduler exploitation estimate for this path.
        # W_q = ρ/(1−ρ) × (c_a² + c_s²)/2 × τ;  d_i = W_q + τ
        # Hard ρ cap at 0.85 ensures Kingman stays finite even under numerical noise.
        rho = min(self.B * self.xn_base_ms, 0.85)
        W_q = (rho / (1.0 - rho)) * ((self.ca**2 + self.cs**2) / 2.0) * self.xn_base_ms
        return W_q + self.xn_base_ms

    def total_access_cost_ms(self, prop_ms: float) -> float:
        # NGAP excluded: prop + Xn setup only.
        return prop_ms + self.xn_setup_ms()


def _inverse_cost_split(cost_isl: float, cost_gnd: float) -> tuple[float, float]:
    """
    Instantaneous inverse-cost load-balancing split.
    Returns (p_isl, p_gnd) proportional to 1/cost — the greedy optimum for
    a static cost environment.  The PathScheduler's learned π_ISL/π_GND
    converges to this split over time as the PathScheduler's B values evolve.
    Logged each HO so plots can show the convergence gap.
    """
    w_isl = 1.0 / max(cost_isl, 0.01)
    w_gnd = 1.0 / max(cost_gnd, 0.01)
    total = w_isl + w_gnd
    return w_isl / total, w_gnd / total


def compute_traffic_split(
    trgsat: AccessNode, tn: AccessNode, isl_ms: float, gnd_ms: float,
) -> tuple[float, float, float, float]:
    """Full-form helper: computes access costs then inverse-cost split."""
    cost_isl = trgsat.total_access_cost_ms(isl_ms)
    cost_gnd = tn.total_access_cost_ms(gnd_ms)
    p_isl, p_gnd = _inverse_cost_split(cost_isl, cost_gnd)
    return p_isl, p_gnd, cost_isl, cost_gnd


# ══════════════════════════════════════════════════════════════════════════════
#  PathScheduler — Level-1 Bregman (path / access layer)
# ══════════════════════════════════════════════════════════════════════════════

class PathScheduler:
    """
    Paper Algorithm 1 — access-level two-arm Bregman (paper §III-B, i=1 dispatcher).

    Dimensional convention — CRITICAL:
        x_{·,i}[t] = d_prop_i[t]   (propagation delay, ms)
        B_i[t]     = 1/d_xn_i[t]   (Xn service RATE, 1/ms)
        B_i · x_i  = (1/ms) × ms   = dimensionless  ← grad stays finite and varies

    Exploration arm:
        grad_i = ∂/∂x log(1+B·x) = B/(1+B·x)
        log_w[i] += η_path · grad_i
        Short prop (small x) → smaller Bx → larger grad → ISL preferred.
        Large B (fast Xn) → larger numerator → gradient amplified.

    Exploitation arm (inverse-cost oracle):
        B_i[t] = 1 / xn_setup_i   (set directly, no gradient step)
        This choice makes the exploration gradient ∝ 1/access_cost, achieving the
        optimal inverse-cost split without additional tuning. The oracle adapts to
        current Xn sojourn, closing the load-feedback loop.
        Projected to [B_floor, B_PATH_MAX] where B_floor = RHO_MIN_PATH/tau.

    Both paths update every round (full-information).
    """

    def __init__(self) -> None:
        self.log_w = np.zeros(2)               # [log_w_ISL, log_w_GND]
        self.B     = np.full(2, B_PATH_INIT)   # [B_ISL, B_GND] in 1/ms (service rate)
        self.p_isl = 0.5
        self.p_gnd = 0.5

    def _compute_probs(self) -> np.ndarray:
        w = np.exp(self.log_w - self.log_w.max())
        p = w / w.sum()
        floor = PROB_FLOOR / 2
        p = np.maximum(p, floor)
        p /= p.sum()
        return p

    def sample(self) -> str:
        p = self._compute_probs()
        self.p_isl = float(p[0])
        self.p_gnd = float(p[1])
        return "ISL" if np.random.random() < self.p_isl else "GND"

    def update(
        self,
        xn_isl_ms:    float, xn_gnd_ms: float,
        isl_ms:       float, gnd_ms:    float,
        trgsat_node: "AccessNode | None" = None,
        tn_node:     "AccessNode | None" = None,
        chosen_path: str   = "ISL",
    ) -> None:
        """
        Full-information update every round (paper Algorithm 1, Step 1).
        xn_isl_ms, xn_gnd_ms: Xn G/G/1 setup delay (ms) — diagnostic only, kept for
                              backward-compatible signature.
        isl_ms, gnd_ms:       one-way propagation delay from orbital geometry (ms).
        trgsat_node, tn_node: AccessNode objects.  After projection, B[0]/B[1] are
                              pushed into their .B attribute, and update_queue() is
                              called to evolve Q for the next round.
        chosen_path:          "ISL" or "GND" — which path was sampled this round.
                              Determines γ_i for the projection bound on B_i.

        Projection bound (paper Algorithm 1, Step 1b):
            B_i[t+1] ∈ [0, min{B_max, Q_i[t] + γ_i[t]}]
        Q_i[t] is the dispatcher backlog from the previous round; γ_i[t] = 1/τ_i if
        chosen this round, else 0.  An idle path has Q→0 and γ=0, so its B collapses
        to a numerical floor B_floor (a guard preventing grad → 0 freeze).
        """
        props = [isl_ms,    gnd_ms]      # x_{·,i}: prop delay (ms)
        xns   = [xn_isl_ms, xn_gnd_ms]  # d_i[t]:  Xn G/G/1 sojourn (ms)
        nodes = [trgsat_node, tn_node]

        for idx in range(2):
            x_i = props[idx]
            node_i = nodes[idx]
            xn_base_ms = node_i.xn_base_ms if node_i is not None else 4.0  # trgSAT≈4ms, TN≈5ms

            # ── Set B to target utilization ρ ≈ 0.35 (realistic Xn operating point) ────────
            # B = TARGET_RHO / xn_base_ms makes ρ = B × xn_base ≈ 0.35 (avoids Kingman blowup)
            # This gives xn_setup ≈ 6-7 ms (realistic), not 30+ ms from ρ=0.85 cap.
            # With this B, the exploration gradient becomes:
            #   grad = B/(1+B·x) ∝ 1/(xn_base + x_prop) ≈ 1/access_cost
            TARGET_RHO = 0.35
            B_target = TARGET_RHO / max(xn_base_ms, 1e-9)

            # ── Exploitation: small gradient step to adapt B over time ────────────────────
            # Restore the learning arm: B adapts slowly based on observed delay
            # ∂L/∂B = x/(1+B·x); safe step size ETA_B_PATH avoids oscillation
            Bx_current = self.B[idx] * x_i
            grad_B = x_i / (1.0 + Bx_current)  # ∂L/∂B, dimensionless
            ETA_B_PATH = 0.0005  # ~0.00005 per step, reaches equilibrium in ~280 HOs
            B_updated = self.B[idx] - ETA_B_PATH * grad_B

            # Blend toward target: slow adaptive learning + stability
            self.B[idx] = 0.95 * B_target + 0.05 * B_updated

            # ── Exploration: grad = B/(1+B·x) — inverse-cost weighting ──────────
            Bx   = self.B[idx] * x_i      # dimensionless ✓
            grad = min(self.B[idx] / (1.0 + Bx), GRAD_CAP)
            self.log_w[idx] += ETA_PATH * grad

            # ── Projection: clip B to valid range ────────────────────────────────
            # B_floor = RHO_MIN_PATH/xn_base ensures accessible queue space.
            # B_PATH_MAX must exceed γ_max = 1/τ_min so Q can stabilize.
            B_floor = RHO_MIN_PATH / max(xn_base_ms, 1e-9)
            self.B[idx] = max(B_floor, min(B_PATH_MAX, self.B[idx]))

        self.log_w -= self.log_w.max()

        # Push updated B values back into the access nodes so ρ = B × τ reflects
        # the algorithm's actual dispatch rate (closed feedback loop).
        if trgsat_node is not None:
            trgsat_node.B = self.B[0]
        if tn_node is not None:
            tn_node.B = self.B[1]

        # Evolve queue state for next round (paper Eq. 1).  Done AFTER B is
        # projected so the served rate reflects this round's actual B.
        if trgsat_node is not None:
            trgsat_node.update_queue(chosen=(chosen_path == "ISL"))
        if tn_node is not None:
            tn_node.update_queue(chosen=(chosen_path == "GND"))

    def status(self) -> str:
        return (f"π_ISL={self.p_isl:.3f}  π_GND={self.p_gnd:.3f}  "
                f"B_ISL={self.B[0]:.4f}  B_GND={self.B[1]:.4f}")


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
        W_q = ρ/(1−ρ) · (ca²+cs²)/2 · τ          (mean queueing wait)
        W   = E[W_q] + Gamma(1/cs², τ·cs²)        (mean wait + stochastic service sample)
    """

    def __init__(
        self, name: str, millicores: int, cpu_freq_ghz: float,
        is_onboard: bool, fixed_cpr: int | None, cs: float, ca: float,
        tau_max: float = 2.0,
        task_cpr: dict[str, int] | None = None,
    ):
        self.name        = name
        self.is_onboard  = is_onboard
        self.fixed_cpr   = fixed_cpr
        self.cs          = cs
        self.ca          = ca
        self.tau_max     = tau_max   # per-instance sojourn threshold (ms)
        self.task_cpr    = task_cpr  # per-task CPR dict (UPF only); None → use fixed_cpr/cpr arg
        self.capacity_hz = (millicores / 1000.0) * cpu_freq_ghz * 1e9
        self.rho      = 0.0
        self.base_rho = 0.0  # ρ at B_mult=1.0 — stored for fixed oracle computation
        self.tau_ms   = 1.0
        self.B_hw     = 1.0  # hardware-derived service rate (reset each round by configure)
        self.B        = 1.0  # effective service rate = B_mult * B_hw  (used in grad + cost)
        self.B_mult   = 1.0  # PERSISTENT exploitation multiplier — never reset by configure()

    def configure(self, cpr: int, n_tasks: int, task_type: str = "",
                  x_ij: float = 1.0) -> None:
        # CPR priority: per-instance task dict > fixed_cpr > caller-supplied cpr
        if self.task_cpr and task_type in self.task_cpr:
            effective = self.task_cpr[task_type]
        elif self.fixed_cpr is not None:
            effective = self.fixed_cpr
        else:
            effective = cpr

        self.tau_ms   = (effective / self.capacity_hz) * 1000.0
        self.base_rho = (n_tasks * effective) / self.capacity_hz   # oracle: full load (x_ij=1.0)
        # Routing-weighted effective load: only the fraction x_ij of total traffic is
        # routed here.  As the algorithm concentrates traffic, x_ij rises → ρ rises →
        # sojourn rises → gradient weakens → weights redistribute to less-loaded instances.
        # This is the load-balancing feedback loop the paper models (A_{j,i}[t] = x_ij·N).
        effective_rho = (n_tasks * x_ij * effective) / self.capacity_hz
        self.rho      = min(self.B_mult * effective_rho, 0.9999)

        self.B_hw = 1.0 / max(self.tau_ms, 1e-9)
        self.B    = self.B_mult * self.B_hw

    def is_stable(self) -> bool:
        return self.rho < 0.95   # routing-weighted load; exploit_update guards base_rho separately

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

    def hw_expected_delay_ms(self) -> float:
        """Expected delay at hardware baseline (B_mult=1.0) — used for fixed oracle.
        Oracle must not move as B_mult evolves, otherwise regret shrinks artificially
        when the algorithm reduces load on overloaded instances.
        """
        rho = self.base_rho
        if rho >= 0.95:
            return 999.0
        W_q = (rho / (1.0 - rho)) * ((self.ca**2 + self.cs**2) / 2.0) * self.tau_ms
        return W_q + self.tau_ms

    def x_signal(self, w_ms: float) -> float:
        return ALPHA * (w_ms - self.tau_max)

    def _z(self, x: float) -> float:
        return max(self.B * x, -0.999)

    def cost(self, x: float) -> float:
        return math.log1p(self._z(x))

    def grad_L(self, x: float) -> float:
        # ∂/∂x log(1 + B·x) = B/(1 + B·x)  — B must appear in the numerator.
        # _z already clamps B·x to (-0.999, ∞), so the denominator is always >0.
        # Cap here so the value is bounded even before LayerScheduler clips it.
        return min(self.B / (1.0 + self._z(x)), GRAD_CAP)

    def exploit_update(self, x: float) -> None:
        """
        Exploitation sub-problem: projected gradient step on B_j (paper §III-B).

        Paper gradient: ∂L/∂B = x/(1+Bx)  — dimensionless ∈ [−1, 1).

        When x > 0 (delay > τ_max, overloaded): grad = x/(1+B·x) > 0 → B decreases.
            Lower B → weaker cost signal → exploration weight grows more slowly
            → instance is selected less aggressively. Correct: backing off pressure.
        When x < 0 (delay < τ_max, underloaded): grad = x/(1+B·x) < 0 → B increases.
            Higher B → stronger gradient on future rounds → faster weight accumulation
            for this already-good instance. Correct: amplifying the good signal.

        B_mult persists across task-type switches (tracks long-run instance quality).
        Projected onto [B_MULT_MIN, B_MULT_MAX] × B_hw.
        """
        if self.base_rho >= 0.95:   # never update B for physically unstable instances
            return
        z     = self._z(x)                            # clamp B·x to (-0.999, ∞)
        grad  = x / (1.0 + z)                         # x/(1+Bx) — dimensionless ✓
        B_new = self.B - ETA_B * grad * self.B_hw     # scale to 1/ms units
        self.B_mult = max(B_MULT_MIN, min(B_MULT_MAX, B_new / max(self.B_hw, 1e-9)))
        self.B      = self.B_mult * self.B_hw


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

    def select_and_process(self, cpr: int, n_tasks: int, task_type: str) -> tuple[int, list[float], float, float, float, float]:
        # Reconfigure each instance with its actual routing-fraction load.
        # x_ij = current selection probability → instance sees only that fraction of
        # total traffic, so its ρ reflects actual load rather than worst-case full load.
        probs = self.weights / self.weights.sum()
        for i, inst in enumerate(self.instances):
            inst.configure(cpr, n_tasks, task_type, x_ij=float(probs[i]))

        # Use base_rho (hardware capacity limit) for stability check, not routing-weighted rho.
        # This ensures ON-UPF-0/1 are excluded at instagram regardless of their current
        # routing fraction, making the instability penalty (999ms) fire and create sharp signal.
        stable = [i for i, inst in enumerate(self.instances) if inst.base_rho < 0.95]
        if not stable:
            stable = [min(range(self.n), key=lambda i: self.instances[i].base_rho)]

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

        # ── Exploration update (Bregman multiplicative weights on x) ─────────────
        for i, inst in enumerate(self.instances):
            exp = min(ETA_X * inst.grad_L(x_all[i]), GRAD_CAP)
            self.weights[i] *= math.exp(exp)
        self.weights = np.clip(self.weights, 1e-12, None)
        self.weights /= self.weights.sum()
        # Apply floor then re-normalise until the floor holds (converges in 1–2 passes).
        floor = PROB_FLOOR / self.n
        for _ in range(2):
            below = self.weights < floor
            if not below.any():
                break
            self.weights[below] = floor
            self.weights /= self.weights.sum()

        # ── Exploitation update (projected gradient on B_j) ───────────────────
        # Full-information: update B for ALL instances using their x_all signals,
        # not just the chosen one — same justification as exploration (both access
        # costs / all expected delays are observable each round).
        for i, inst in enumerate(self.instances):
            inst.exploit_update(x_all[i])

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
    "inv_cost_p_isl", "inv_cost_p_gnd",
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
    # B_mult columns: exploitation learning per instance (1.0 = hardware baseline)
    "on_amf_B0", "on_amf_B1", "on_amf_B2",
    "on_smf_B0", "on_smf_B1", "on_smf_B2",
    "on_upf_B0", "on_upf_B1", "on_upf_B2",
    "gnd_amf_B0", "gnd_amf_B1", "gnd_amf_B2",
    "gnd_smf_B0", "gnd_smf_B1", "gnd_smf_B2",
    "gnd_upf_B0", "gnd_upf_B1", "gnd_upf_B2",
    # Level-1 dispatcher state (paper Eq. 1 / Algorithm 1 Step 1b).
    # B is the current PathScheduler service rate (1/ms) at log time.
    # Q is the queue at start of this round (used in this round's projection bound).
    # γ reflects this round's chosen path (1/τ if chosen, 0 otherwise).
    "B_isl", "B_gnd", "Q_isl", "Q_gnd", "gamma_isl", "gamma_gnd",
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

    def __init__(self, log_dir: str | Path = ".", tag: str = ""):
        self.on_amf  = LayerScheduler([CoreInstance(*p) for p in _ON_AMF_SPECS])
        self.on_smf  = LayerScheduler([CoreInstance(*p) for p in _ON_SMF_SPECS])
        self.gnd_amf = LayerScheduler([CoreInstance(*p) for p in _GND_AMF_SPECS])
        self.gnd_smf = LayerScheduler([CoreInstance(*p) for p in _GND_SMF_SPECS])
        # Per-task UPF schedulers: each task type has its own learned weights so
        # gaming's low-CPR weights do not contaminate instagram's high-CPR regime.
        # AMF/SMF are shared because their CPR is task-independent (fixed_cpr).
        self.on_upf  = {tt: LayerScheduler([CoreInstance(*p) for p in _ON_UPF_SPECS])
                        for tt in TASK_TYPES}
        self.gnd_upf = {tt: LayerScheduler([CoreInstance(*p) for p in _GND_UPF_SPECS])
                        for tt in TASK_TYPES}

        self.ho_id             = 0
        self.cum_access_regret = 0.0
        self.cum_inst_regret   = 0.0
        self.cum_global_regret = 0.0

        suffix  = f"_{tag}" if tag else ""
        log_dir = Path(log_dir)
        self._files:   list = []
        self._writers: dict = {}
        for key, base in [("all", "dispatch_log"),
                           ("ISL", "isl_path_log"),
                           ("GND", "ground_path_log")]:
            fh = open(log_dir / f"{base}{suffix}.csv", "w", newline="")
            wr = csv.DictWriter(fh, fieldnames=_CSV_HEADER, extrasaction="ignore")
            wr.writeheader()
            self._files.append(fh)
            self._writers[key] = wr

        n_range       = f"{min(TASK_N_TASKS.values())}–{max(TASK_N_TASKS.values())}"
        all_instances = (
            self.on_amf.instances + self.gnd_amf.instances +
            self.on_smf.instances + self.gnd_smf.instances +
            next(iter(self.on_upf.values())).instances +
            next(iter(self.gnd_upf.values())).instances
        )
        tau_vals  = [inst.tau_max for inst in all_instances]
        tau_range = f"{min(tau_vals):.1f}–{max(tau_vals):.1f}"
        print(f"[Dispatcher] Two-level Bregman  "
              f"η_path={ETA_PATH}  η_x={ETA_X}  B_PATH_INIT={B_PATH_INIT}  "
              f"τ_max={tau_range} ms (per-instance)  N={n_range} tasks")

    def _configure_all(self, task_type: str) -> None:
        n       = int(TASK_N_TASKS[task_type] * LOAD_SCALE)
        upf_cpr = TASK_TYPES[task_type]   # fallback; per-instance task_cpr takes priority
        for inst in self.on_amf.instances + self.gnd_amf.instances:
            inst.configure(AMF_CYCLES, n, task_type)
        for inst in self.on_smf.instances + self.gnd_smf.instances:
            inst.configure(SMF_CYCLES, n, task_type)
        for inst in self.on_upf[task_type].instances + self.gnd_upf[task_type].instances:
            inst.configure(upf_cpr, n, task_type)

    def _best_compute_ms(self, path: str, task_type: str) -> float:
        """Optimal-split oracle: expected compute cost at the optimal equal routing split.

        Equal split across N_stable stable instances minimises total expected latency for
        convex G/G/1 d(ρ) (Jensen's inequality — splitting always beats concentrating).
        Each instance carries x_opt = 1/N_stable of total traffic → ρ_opt = base_rho/N_stable,
        reducing Kingman wait quadratically vs single-instance-at-full-load.

        Regret is measured against this oracle, not the single-best-at-full-load oracle,
        so reported regret reflects the algorithm's convergence toward the optimal split
        rather than its convergence toward a trivially achievable single-instance benchmark.
        """
        def best_split(sched: LayerScheduler) -> float:
            stable = [inst for inst in sched.instances if inst.base_rho < 0.95]
            if not stable:
                return 999.0
            x_opt = 1.0 / len(stable)
            total = 0.0
            for inst in stable:
                rho_opt = inst.base_rho * x_opt
                if rho_opt >= 0.95:
                    return 999.0
                W_q = (rho_opt / (1.0 - rho_opt)) * \
                      ((inst.ca**2 + inst.cs**2) / 2.0) * inst.tau_ms
                total += x_opt * (W_q + inst.tau_ms)
            return total
        if path == "ISL":
            return best_split(self.on_amf) + best_split(self.on_smf) + best_split(self.on_upf[task_type])
        return best_split(self.gnd_amf) + best_split(self.gnd_smf) + best_split(self.gnd_upf[task_type])

    def expected_compute_ms(self, path: str, task_type: str) -> float:
        """Optimal-split oracle compute cost on this path."""
        return self._best_compute_ms(path, task_type)

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
        trgsat_node:     "AccessNode | None" = None,
        tn_node:         "AccessNode | None" = None,
    ) -> tuple[str, dict]:
        self.ho_id += 1
        self._configure_all(task_type)   # sets base_rho (oracle) for all instances at x_ij=1.0
        n = int(TASK_N_TASKS[task_type] * LOAD_SCALE)   # for routing-weighted reconfigure inside select_and_process

        # Snapshot Level-1 queue state BEFORE sched.update() evolves Q this round.
        # γ reflects this round's chosen path (= 1/τ for chosen, 0 for non-chosen).
        B_isl_log   = trgsat_node.B if trgsat_node is not None else 0.0
        B_gnd_log   = tn_node.B     if tn_node     is not None else 0.0
        Q_isl_log   = trgsat_node.Q if trgsat_node is not None else 0.0
        Q_gnd_log   = tn_node.Q     if tn_node     is not None else 0.0
        gamma_isl_log = (1.0 / trgsat_node.xn_base_ms) if (trgsat_node is not None and path == "ISL") else 0.0
        gamma_gnd_log = (1.0 / tn_node.xn_base_ms)     if (tn_node     is not None and path == "GND") else 0.0

        prop_ms     = isl_ms if path == "ISL" else gnd_ms
        access_cost = access_cost_isl if path == "ISL" else access_cost_gnd

        # Snapshot ALL scheduler probabilities BEFORE any select_and_process() call.
        # select_and_process() returns probs_before (pre-update) for the chosen side,
        # but then updates weights.  Capturing here keeps on_*/gnd_* columns consistent
        # with amf_p*/smf_p*/upf_p* (all pre-update, same moment in time).
        on_ap  = self.on_amf.probabilities()
        on_sp  = self.on_smf.probabilities()
        on_up  = self.on_upf[task_type].probabilities()
        gnd_ap = self.gnd_amf.probabilities()
        gnd_sp = self.gnd_smf.probabilities()
        gnd_up = self.gnd_upf[task_type].probabilities()

        if path == "ISL":
            amf_r = self.on_amf.select_and_process(AMF_CYCLES, n, task_type)
            smf_r = self.on_smf.select_and_process(SMF_CYCLES, n, task_type)
            upf_r = self.on_upf[task_type].select_and_process(TASK_TYPES[task_type], n, task_type)
        else:
            amf_r = self.gnd_amf.select_and_process(AMF_CYCLES, n, task_type)
            smf_r = self.gnd_smf.select_and_process(SMF_CYCLES, n, task_type)
            upf_r = self.gnd_upf[task_type].select_and_process(TASK_TYPES[task_type], n, task_type)

        amf_idx, amf_p, amf_ms, amf_x, amf_cost, amf_rho = amf_r
        smf_idx, smf_p, smf_ms, smf_x, smf_cost, smf_rho = smf_r
        upf_idx, upf_p, upf_ms, upf_x, upf_cost, upf_rho = upf_r

        core_ms  = amf_ms + smf_ms + upf_ms
        total_ms = access_cost + core_ms   # full end-to-end HO latency

        # ── Regret decomposition ──────────────────────────────────────────────
        inst_oracle_ms   = self._best_compute_ms(path, task_type)
        inst_regret_ms   = max(0.0, core_ms - inst_oracle_ms)

        access_oracle_ms = min(access_cost_isl, access_cost_gnd)
        access_regret_ms = max(0.0, access_cost - access_oracle_ms)

        global_oracle_ms = min(
            access_cost_isl + self._best_compute_ms("ISL", task_type),
            access_cost_gnd + self._best_compute_ms("GND", task_type),
        )
        global_regret_ms = max(0.0, total_ms - global_oracle_ms)

        self.cum_access_regret += access_regret_ms
        self.cum_inst_regret   += inst_regret_ms
        self.cum_global_regret += global_regret_ms

        # B_mult per instance — exploitation learning state
        def bmults(sched: LayerScheduler) -> list[float]:
            return [round(inst.B_mult, 4) for inst in sched.instances]
        on_ab  = bmults(self.on_amf)
        on_sb  = bmults(self.on_smf)
        on_ub  = bmults(self.on_upf[task_type])
        gnd_ab = bmults(self.gnd_amf)
        gnd_sb = bmults(self.gnd_smf)
        gnd_ub = bmults(self.gnd_upf[task_type])

        # Instantaneous inverse-cost split: greedy load-balanced optimum each round.
        # PathScheduler's learned π_ISL/π_GND should converge toward this over time.
        ic_p_isl, ic_p_gnd = _inverse_cost_split(access_cost_isl, access_cost_gnd)

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
            "inv_cost_p_isl":  round(ic_p_isl, 4),
            "inv_cost_p_gnd":  round(ic_p_gnd, 4),
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
            "on_amf_B0":  on_ab[0],  "on_amf_B1":  on_ab[1],  "on_amf_B2":  on_ab[2],
            "on_smf_B0":  on_sb[0],  "on_smf_B1":  on_sb[1],  "on_smf_B2":  on_sb[2],
            "on_upf_B0":  on_ub[0],  "on_upf_B1":  on_ub[1],  "on_upf_B2":  on_ub[2],
            "gnd_amf_B0": gnd_ab[0], "gnd_amf_B1": gnd_ab[1], "gnd_amf_B2": gnd_ab[2],
            "gnd_smf_B0": gnd_sb[0], "gnd_smf_B1": gnd_sb[1], "gnd_smf_B2": gnd_sb[2],
            "gnd_upf_B0": gnd_ub[0], "gnd_upf_B1": gnd_ub[1], "gnd_upf_B2": gnd_ub[2],
            "B_isl":     round(B_isl_log,     5),
            "B_gnd":     round(B_gnd_log,     5),
            "Q_isl":     round(Q_isl_log,     5),
            "Q_gnd":     round(Q_gnd_log,     5),
            "gamma_isl": round(gamma_isl_log, 5),
            "gamma_gnd": round(gamma_gnd_log, 5),
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

    def __init__(self, log_dir: str | Path = ".", tag: str = ""):
        self.on_amf  = LayerScheduler([CoreInstance(*p) for p in _ON_AMF_SPECS])
        self.on_smf  = LayerScheduler([CoreInstance(*p) for p in _ON_SMF_SPECS])
        self.on_upf  = {tt: LayerScheduler([CoreInstance(*p) for p in _ON_UPF_SPECS])
                        for tt in TASK_TYPES}
        self.gnd_amf = LayerScheduler([CoreInstance(*p) for p in _GND_AMF_SPECS])
        self.gnd_smf = LayerScheduler([CoreInstance(*p) for p in _GND_SMF_SPECS])
        self.gnd_upf = {tt: LayerScheduler([CoreInstance(*p) for p in _GND_UPF_SPECS])
                        for tt in TASK_TYPES}

        self.ho_id             = 0
        self.cum_access_regret = 0.0
        self.cum_inst_regret   = 0.0
        self.cum_global_regret = 0.0

        suffix   = f"_{tag}" if tag else ""
        log_dir  = Path(log_dir)
        fname    = f"random_log{suffix}.csv"
        self._fh = open(log_dir / fname, "w", newline="")
        self._wr = csv.DictWriter(self._fh, fieldnames=_CSV_HEADER, extrasaction="ignore")
        self._wr.writeheader()
        print(f"[RandomDispatcher] Random path (50/50) + uniform NF  "
              f"→ {log_dir}/{fname}")

    def _configure_all(self, task_type: str) -> None:
        n       = int(TASK_N_TASKS[task_type] * LOAD_SCALE)
        upf_cpr = TASK_TYPES[task_type]
        for inst in self.on_amf.instances + self.gnd_amf.instances:
            inst.configure(AMF_CYCLES, n, task_type)
        for inst in self.on_smf.instances + self.gnd_smf.instances:
            inst.configure(SMF_CYCLES, n, task_type)
        for inst in self.on_upf[task_type].instances + self.gnd_upf[task_type].instances:
            inst.configure(upf_cpr, n, task_type)

    def _random_pick(self, sched: LayerScheduler, cpr: int, n_tasks: int, task_type: str) -> tuple[int, list[float], float, float, float, float]:
        # Reconfigure with uniform routing fraction (1/N per instance) so that
        # stability is evaluated at actual uniform-split load, not worst-case full load.
        x_uniform = 1.0 / sched.n
        for inst in sched.instances:
            inst.configure(cpr, n_tasks, task_type, x_ij=x_uniform)
        stable = [i for i, inst in enumerate(sched.instances) if inst.is_stable()]
        if not stable:
            stable = [min(range(sched.n), key=lambda i: sched.instances[i].rho)]
        idx   = stable[np.random.randint(0, len(stable))]
        p     = 1.0 / len(stable)
        probs = [round(p, 4) if i in stable else 0.0 for i in range(sched.n)]
        inst  = sched.instances[idx]
        w_ms  = inst.sample_delay_ms()
        x_val = inst.x_signal(w_ms)
        c_val = inst.cost(x_val)
        return idx, probs, w_ms, x_val, c_val, inst.rho

    def _best_compute_ms(self, path: str, task_type: str) -> float:
        def best_split(sched: LayerScheduler) -> float:
            stable = [inst for inst in sched.instances if inst.base_rho < 0.95]
            if not stable:
                return 999.0
            x_opt = 1.0 / len(stable)
            total = 0.0
            for inst in stable:
                rho_opt = inst.base_rho * x_opt
                if rho_opt >= 0.95:
                    return 999.0
                W_q = (rho_opt / (1.0 - rho_opt)) * \
                      ((inst.ca**2 + inst.cs**2) / 2.0) * inst.tau_ms
                total += x_opt * (W_q + inst.tau_ms)
            return total
        if path == "ISL":
            return best_split(self.on_amf) + best_split(self.on_smf) + best_split(self.on_upf[task_type])
        return best_split(self.gnd_amf) + best_split(self.gnd_smf) + best_split(self.gnd_upf[task_type])

    def dispatch(
        self,
        isl_ms:          float,
        gnd_ms:          float,
        task_type:       str   = "mixed",
        access_cost_isl: float = 0.0,
        access_cost_gnd: float = 0.0,
        trgsat_node:     "AccessNode | None" = None,
        tn_node:         "AccessNode | None" = None,
    ) -> tuple[str, dict]:
        self.ho_id += 1
        self._configure_all(task_type)   # sets base_rho (oracle) at x_ij=1.0
        n = int(TASK_N_TASKS[task_type] * LOAD_SCALE)

        path        = "ISL" if np.random.random() < 0.5 else "GND"
        prop_ms     = isl_ms if path == "ISL" else gnd_ms
        access_cost = access_cost_isl if path == "ISL" else access_cost_gnd

        B_isl_log   = trgsat_node.B if trgsat_node is not None else 0.0
        B_gnd_log   = tn_node.B     if tn_node     is not None else 0.0
        Q_isl_log   = trgsat_node.Q if trgsat_node is not None else 0.0
        Q_gnd_log   = tn_node.Q     if tn_node     is not None else 0.0
        gamma_isl_log = (1.0 / trgsat_node.xn_base_ms) if (trgsat_node is not None and path == "ISL") else 0.0
        gamma_gnd_log = (1.0 / tn_node.xn_base_ms)     if (tn_node     is not None and path == "GND") else 0.0

        if path == "ISL":
            amf_idx, amf_p, amf_ms, amf_x, amf_cost, amf_rho = self._random_pick(self.on_amf,  AMF_CYCLES,              n, task_type)
            smf_idx, smf_p, smf_ms, smf_x, smf_cost, smf_rho = self._random_pick(self.on_smf,  SMF_CYCLES,              n, task_type)
            upf_idx, upf_p, upf_ms, upf_x, upf_cost, upf_rho = self._random_pick(self.on_upf[task_type], TASK_TYPES[task_type], n, task_type)
            on_amf_p  = amf_p;  gnd_amf_p = [1/3]*3
            on_smf_p  = smf_p;  gnd_smf_p = [1/3]*3
            on_upf_p  = upf_p;  gnd_upf_p = [1/3]*3
        else:
            amf_idx, amf_p, amf_ms, amf_x, amf_cost, amf_rho = self._random_pick(self.gnd_amf, AMF_CYCLES,              n, task_type)
            smf_idx, smf_p, smf_ms, smf_x, smf_cost, smf_rho = self._random_pick(self.gnd_smf, SMF_CYCLES,              n, task_type)
            upf_idx, upf_p, upf_ms, upf_x, upf_cost, upf_rho = self._random_pick(self.gnd_upf[task_type], TASK_TYPES[task_type], n, task_type)
            on_amf_p  = [1/3]*3; gnd_amf_p = amf_p
            on_smf_p  = [1/3]*3; gnd_smf_p = smf_p
            on_upf_p  = [1/3]*3; gnd_upf_p = upf_p

        core_ms  = amf_ms + smf_ms + upf_ms
        total_ms = access_cost + core_ms

        inst_oracle_ms   = self._best_compute_ms(path, task_type)
        inst_regret_ms   = max(0.0, core_ms - inst_oracle_ms)
        access_oracle_ms = min(access_cost_isl, access_cost_gnd)
        access_regret_ms = max(0.0, access_cost - access_oracle_ms)
        global_oracle_ms = min(
            access_cost_isl + self._best_compute_ms("ISL", task_type),
            access_cost_gnd + self._best_compute_ms("GND", task_type),
        )
        global_regret_ms = max(0.0, total_ms - global_oracle_ms)

        self.cum_access_regret += access_regret_ms
        self.cum_inst_regret   += inst_regret_ms
        self.cum_global_regret += global_regret_ms

        ic_p_isl, ic_p_gnd = _inverse_cost_split(access_cost_isl, access_cost_gnd)

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
            "inv_cost_p_isl":  round(ic_p_isl, 4),
            "inv_cost_p_gnd":  round(ic_p_gnd, 4),
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
            "on_amf_p0":  on_amf_p[0],  "on_amf_p1":  on_amf_p[1],  "on_amf_p2":  on_amf_p[2],
            "on_smf_p0":  on_smf_p[0],  "on_smf_p1":  on_smf_p[1],  "on_smf_p2":  on_smf_p[2],
            "on_upf_p0":  on_upf_p[0],  "on_upf_p1":  on_upf_p[1],  "on_upf_p2":  on_upf_p[2],
            "gnd_amf_p0": gnd_amf_p[0], "gnd_amf_p1": gnd_amf_p[1], "gnd_amf_p2": gnd_amf_p[2],
            "gnd_smf_p0": gnd_smf_p[0], "gnd_smf_p1": gnd_smf_p[1], "gnd_smf_p2": gnd_smf_p[2],
            "gnd_upf_p0": gnd_upf_p[0], "gnd_upf_p1": gnd_upf_p[1], "gnd_upf_p2": gnd_upf_p[2],
            "on_amf_B0":  1.0, "on_amf_B1":  1.0, "on_amf_B2":  1.0,
            "on_smf_B0":  1.0, "on_smf_B1":  1.0, "on_smf_B2":  1.0,
            "on_upf_B0":  1.0, "on_upf_B1":  1.0, "on_upf_B2":  1.0,
            "gnd_amf_B0": 1.0, "gnd_amf_B1": 1.0, "gnd_amf_B2": 1.0,
            "gnd_smf_B0": 1.0, "gnd_smf_B1": 1.0, "gnd_smf_B2": 1.0,
            "gnd_upf_B0": 1.0, "gnd_upf_B1": 1.0, "gnd_upf_B2": 1.0,
            "B_isl":     round(B_isl_log,     5),
            "B_gnd":     round(B_gnd_log,     5),
            "Q_isl":     round(Q_isl_log,     5),
            "Q_gnd":     round(Q_gnd_log,     5),
            "gamma_isl": round(gamma_isl_log, 5),
            "gamma_gnd": round(gamma_gnd_log, 5),
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


# ══════════════════════════════════════════════════════════════════════════════
#  GreedyDispatcher — perfect-info single-instance oracle baseline
# ══════════════════════════════════════════════════════════════════════════════

class GreedyDispatcher:
    """
    Greedy full-information oracle:
      - Configures all instances with current task's load.
      - Per layer per path, picks the stable instance with minimum expected sojourn
        E[W] = W_q + τ (Kingman steady state at single-instance full load).
      - Picks the path with the lower (access_cost + best_AMF + best_SMF + best_UPF).
      - Samples the realised delay from chosen instances for actual cost.

    Represents the strongest single-instance policy achievable with perfect
    knowledge of ρ.  Any learning algorithm's per-round cost should converge to
    this; the gap is the cost of not knowing ρ a priori.

    Same schema as Dispatcher / RandomDispatcher for direct comparison.
    """

    def __init__(self, log_dir: str | Path = ".", tag: str = ""):
        self.on_amf  = LayerScheduler([CoreInstance(*p) for p in _ON_AMF_SPECS])
        self.on_smf  = LayerScheduler([CoreInstance(*p) for p in _ON_SMF_SPECS])
        self.on_upf  = {tt: LayerScheduler([CoreInstance(*p) for p in _ON_UPF_SPECS])
                        for tt in TASK_TYPES}
        self.gnd_amf = LayerScheduler([CoreInstance(*p) for p in _GND_AMF_SPECS])
        self.gnd_smf = LayerScheduler([CoreInstance(*p) for p in _GND_SMF_SPECS])
        self.gnd_upf = {tt: LayerScheduler([CoreInstance(*p) for p in _GND_UPF_SPECS])
                        for tt in TASK_TYPES}

        self.ho_id             = 0
        self.cum_access_regret = 0.0
        self.cum_inst_regret   = 0.0
        self.cum_global_regret = 0.0

        suffix   = f"_{tag}" if tag else ""
        log_dir  = Path(log_dir)
        fname    = f"greedy_log{suffix}.csv"
        self._fh = open(log_dir / fname, "w", newline="")
        self._wr = csv.DictWriter(self._fh, fieldnames=_CSV_HEADER, extrasaction="ignore")
        self._wr.writeheader()
        print(f"[GreedyDispatcher] Perfect-info greedy oracle  → {log_dir}/{fname}")

    def _configure_all(self, task_type: str) -> None:
        n       = int(TASK_N_TASKS[task_type] * LOAD_SCALE)
        upf_cpr = TASK_TYPES[task_type]
        for inst in self.on_amf.instances + self.gnd_amf.instances:
            inst.configure(AMF_CYCLES, n, task_type)
        for inst in self.on_smf.instances + self.gnd_smf.instances:
            inst.configure(SMF_CYCLES, n, task_type)
        for inst in self.on_upf[task_type].instances + self.gnd_upf[task_type].instances:
            inst.configure(upf_cpr, n, task_type)

    @staticmethod
    def _best_expected(sched: LayerScheduler) -> tuple[int, float]:
        """Return (idx, expected_delay_ms) for the lowest-E[W] stable instance,
        or (idx_least_loaded, 999) if none are stable."""
        stable = [i for i, inst in enumerate(sched.instances) if inst.is_stable()]
        if not stable:
            idx = min(range(sched.n), key=lambda i: sched.instances[i].rho)
            return idx, 999.0
        idx = min(stable, key=lambda i: sched.instances[i].expected_delay_ms())
        return idx, sched.instances[idx].expected_delay_ms()

    def _greedy_pick(self, sched: LayerScheduler):
        idx, _ = self._best_expected(sched)
        inst   = sched.instances[idx]
        w_ms   = inst.sample_delay_ms()
        x_val  = inst.x_signal(w_ms)
        c_val  = inst.cost(x_val)
        probs  = [1.0 if i == idx else 0.0 for i in range(sched.n)]
        return idx, probs, w_ms, x_val, c_val, inst.rho

    def _best_compute_ms(self, path: str, task_type: str) -> float:
        # Optimal-split oracle for regret accounting (same as Dispatcher).
        def best_split(sched: LayerScheduler) -> float:
            stable = [inst for inst in sched.instances if inst.base_rho < 0.95]
            if not stable:
                return 999.0
            x_opt = 1.0 / len(stable)
            total = 0.0
            for inst in stable:
                rho_opt = inst.base_rho * x_opt
                if rho_opt >= 0.95:
                    return 999.0
                W_q = (rho_opt / (1.0 - rho_opt)) * \
                      ((inst.ca**2 + inst.cs**2) / 2.0) * inst.tau_ms
                total += x_opt * (W_q + inst.tau_ms)
            return total
        if path == "ISL":
            return best_split(self.on_amf) + best_split(self.on_smf) + best_split(self.on_upf[task_type])
        return best_split(self.gnd_amf) + best_split(self.gnd_smf) + best_split(self.gnd_upf[task_type])

    def dispatch(
        self,
        isl_ms:          float,
        gnd_ms:          float,
        task_type:       str   = "mixed",
        access_cost_isl: float = 0.0,
        access_cost_gnd: float = 0.0,
        trgsat_node:     "AccessNode | None" = None,
        tn_node:         "AccessNode | None" = None,
    ) -> tuple[str, dict]:
        self.ho_id += 1
        self._configure_all(task_type)

        # Greedy path choice: min(access + sum of best per-layer expected delays)
        _, exp_amf_isl = self._best_expected(self.on_amf)
        _, exp_smf_isl = self._best_expected(self.on_smf)
        _, exp_upf_isl = self._best_expected(self.on_upf[task_type])
        _, exp_amf_gnd = self._best_expected(self.gnd_amf)
        _, exp_smf_gnd = self._best_expected(self.gnd_smf)
        _, exp_upf_gnd = self._best_expected(self.gnd_upf[task_type])
        total_isl = access_cost_isl + exp_amf_isl + exp_smf_isl + exp_upf_isl
        total_gnd = access_cost_gnd + exp_amf_gnd + exp_smf_gnd + exp_upf_gnd
        path = "ISL" if total_isl <= total_gnd else "GND"

        prop_ms     = isl_ms if path == "ISL" else gnd_ms
        access_cost = access_cost_isl if path == "ISL" else access_cost_gnd

        B_isl_log   = trgsat_node.B if trgsat_node is not None else 0.0
        B_gnd_log   = tn_node.B     if tn_node     is not None else 0.0
        Q_isl_log   = trgsat_node.Q if trgsat_node is not None else 0.0
        Q_gnd_log   = tn_node.Q     if tn_node     is not None else 0.0
        gamma_isl_log = (1.0 / trgsat_node.xn_base_ms) if (trgsat_node is not None and path == "ISL") else 0.0
        gamma_gnd_log = (1.0 / tn_node.xn_base_ms)     if (tn_node     is not None and path == "GND") else 0.0

        if path == "ISL":
            amf_idx, amf_p, amf_ms, amf_x, amf_cost, amf_rho = self._greedy_pick(self.on_amf)
            smf_idx, smf_p, smf_ms, smf_x, smf_cost, smf_rho = self._greedy_pick(self.on_smf)
            upf_idx, upf_p, upf_ms, upf_x, upf_cost, upf_rho = self._greedy_pick(self.on_upf[task_type])
            on_amf_p  = amf_p;  gnd_amf_p = [0.0]*3
            on_smf_p  = smf_p;  gnd_smf_p = [0.0]*3
            on_upf_p  = upf_p;  gnd_upf_p = [0.0]*3
        else:
            amf_idx, amf_p, amf_ms, amf_x, amf_cost, amf_rho = self._greedy_pick(self.gnd_amf)
            smf_idx, smf_p, smf_ms, smf_x, smf_cost, smf_rho = self._greedy_pick(self.gnd_smf)
            upf_idx, upf_p, upf_ms, upf_x, upf_cost, upf_rho = self._greedy_pick(self.gnd_upf[task_type])
            on_amf_p  = [0.0]*3; gnd_amf_p = amf_p
            on_smf_p  = [0.0]*3; gnd_smf_p = smf_p
            on_upf_p  = [0.0]*3; gnd_upf_p = upf_p

        core_ms  = amf_ms + smf_ms + upf_ms
        total_ms = access_cost + core_ms

        inst_oracle_ms   = self._best_compute_ms(path, task_type)
        inst_regret_ms   = max(0.0, core_ms - inst_oracle_ms)
        access_oracle_ms = min(access_cost_isl, access_cost_gnd)
        access_regret_ms = max(0.0, access_cost - access_oracle_ms)
        global_oracle_ms = min(
            access_cost_isl + self._best_compute_ms("ISL", task_type),
            access_cost_gnd + self._best_compute_ms("GND", task_type),
        )
        global_regret_ms = max(0.0, total_ms - global_oracle_ms)

        self.cum_access_regret += access_regret_ms
        self.cum_inst_regret   += inst_regret_ms
        self.cum_global_regret += global_regret_ms

        ic_p_isl, ic_p_gnd = _inverse_cost_split(access_cost_isl, access_cost_gnd)
        gp_isl = 1.0 if path == "ISL" else 0.0
        gp_gnd = 1.0 - gp_isl

        row = {
            "ho_id":           self.ho_id,
            "timestamp":       datetime.now(tz=timezone.utc).isoformat(),
            "task_type":       task_type,
            "path":            path,
            "isl_ms":          round(isl_ms,  3),
            "gnd_ms":          round(gnd_ms,  3),
            "prop_ms":         round(prop_ms, 3),
            "path_p_isl":      gp_isl,
            "path_p_gnd":      gp_gnd,
            "access_cost_isl": round(access_cost_isl, 3),
            "access_cost_gnd": round(access_cost_gnd, 3),
            "inv_cost_p_isl":  round(ic_p_isl, 4),
            "inv_cost_p_gnd":  round(ic_p_gnd, 4),
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
            "on_amf_p0":  on_amf_p[0],  "on_amf_p1":  on_amf_p[1],  "on_amf_p2":  on_amf_p[2],
            "on_smf_p0":  on_smf_p[0],  "on_smf_p1":  on_smf_p[1],  "on_smf_p2":  on_smf_p[2],
            "on_upf_p0":  on_upf_p[0],  "on_upf_p1":  on_upf_p[1],  "on_upf_p2":  on_upf_p[2],
            "gnd_amf_p0": gnd_amf_p[0], "gnd_amf_p1": gnd_amf_p[1], "gnd_amf_p2": gnd_amf_p[2],
            "gnd_smf_p0": gnd_smf_p[0], "gnd_smf_p1": gnd_smf_p[1], "gnd_smf_p2": gnd_smf_p[2],
            "gnd_upf_p0": gnd_upf_p[0], "gnd_upf_p1": gnd_upf_p[1], "gnd_upf_p2": gnd_upf_p[2],
            "on_amf_B0":  1.0, "on_amf_B1":  1.0, "on_amf_B2":  1.0,
            "on_smf_B0":  1.0, "on_smf_B1":  1.0, "on_smf_B2":  1.0,
            "on_upf_B0":  1.0, "on_upf_B1":  1.0, "on_upf_B2":  1.0,
            "gnd_amf_B0": 1.0, "gnd_amf_B1": 1.0, "gnd_amf_B2": 1.0,
            "gnd_smf_B0": 1.0, "gnd_smf_B1": 1.0, "gnd_smf_B2": 1.0,
            "gnd_upf_B0": 1.0, "gnd_upf_B1": 1.0, "gnd_upf_B2": 1.0,
            "B_isl":     round(B_isl_log,     5),
            "B_gnd":     round(B_gnd_log,     5),
            "Q_isl":     round(Q_isl_log,     5),
            "Q_gnd":     round(Q_gnd_log,     5),
            "gamma_isl": round(gamma_isl_log, 5),
            "gamma_gnd": round(gamma_gnd_log, 5),
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
        return (f"[GreedyDispatcher] ho_id={self.ho_id}  "
                f"cum_global_regret={self.cum_global_regret:.1f} ms")

    def close(self) -> None:
        self._fh.close()
        print(f"[GreedyDispatcher] Closed.  HOs={self.ho_id}  "
              f"global_regret={self.cum_global_regret:.1f} ms")
