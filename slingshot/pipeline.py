"""Gravitational circuit compiler: chain N flyby gates into one machine.

Ball B launches once and threads N calibrated gravitational gates. Gate k
lives on B's lane at time T_k = k*DT; its control ball C_k flies anti-parallel
to B's incoming lane, offset to the starboard side, timed to arrive at T_k.
A hit bends B's heading one port step (-BEND degrees), so B's final heading
reads out THE INDEX OF THE FIRST ABSENT CONTROL — the machine is at once an
N-input AND (port N) and an N-input priority encoder. The flagship is N=4:
one 5-body machine, 16 input cases, five output ports.

Depth wall (a result, not a bug): this launch-from-afar design tops out near
four gates. A 5th control has to fly ~60 units through the whole accumulated
5-body field to reach B; it is chaotically deflected en route and cannot be
aimed onto B — its calibration Jacobian goes flat (a dead gate). Gates 1-4
compile every time; gate 5 never lands. Reaching deeper needs a different
control architecture (e.g. heavy near-ballistic "mirror" masses). Chaos
bounds not just this machine's PRECISION but its DEPTH.

Compilation is a boundary-value problem, and it is solved like one.

All the bodies form one coupled system: gravity has no shielding, so every
control tugs B and every other control at range (tail deflection ~ 1/b, a
power law). Calibrating gates one at a time is Gauss-Seidel without damping
on a stiff coupled system — it limit-cycles (measured: gate offsets flap
between 0.86 and 1.19 forever, so the next gate aims at a moving target and
misses). The fix is a global solve with step control:

  unknowns  (a_k, p_k) per gate — TWO knobs, both needed: a slide ALONG the
            control's flight line (fixes arrival timing; a control launched
            ~50 units upwind is itself deflected in flight and arrives
            1.8-2.6 time units early, grazing B at ~1.8 instead of ~0.5)
            and a slide along the lane NORMAL (the impact-parameter knob,
            which sets the bend)
  residuals per gate, the LOCAL pair: (bend across THIS encounter minus
            -BEND, arrival-time error), measured in the ALL-ONES
            configuration from ONE sim. The residual must be local: pinning
            cumulative port headings instead lets adjacent gates split a
            port between them (each bending BEND/2 — an exact degenerate
            minimum we hit at 27.0 deg) — the local target forbids it.
  solver    trust-region least squares (scipy least_squares) — its damping
            is exactly what the hand-rolled sweeps lacked

The residual Jacobian is empirically near-lower-triangular (an upstream
slide swings downstream ports ~46 deg/unit, amplified ~x12 per gate;
downstream barely touches upstream) and smooth to 5 significant figures
under finite differencing despite the chaos, and the (timing, bend) pair is
near-diagonal per gate — so the 2Nx2N system is well conditioned and the
solve converges from a cheap greedy forward seed in one shot.

Design point: BEND=54 and unit-speed wires put every encounter at impact
parameter ~1 (tame dtheta/db); DT=12 fans stopped signals ~10 units clear
of downstream control corridors (crosstalk ~ 1/distance, no insulation), so
all 16 mixed-input subsets stay correct on geometric margin — calibration
only ever pins the all-ones lane.
"""

import json
import pathlib

import numpy as np
from scipy.optimize import brentq, least_squares

from . import nbody
from .gates import BALL_B

BEND = 54.0    # heading change per gate, degrees
DT = 12.0      # gate spacing in time
TAIL = 10.0    # coast after the last gate before readout
C_SPEED = 1.0  # control-ball speed
LANE_DT = 6.0  # measure a gate's outgoing lane this long after the encounter
SOLVE_RTOL = 1e-9  # calibration only needs headings to <0.1 deg; the
                   # integrator spends its time resolving encounters to
                   # 1e-12, so relax it ~1000x for the solve and validate
                   # the finished machine at full precision

SPEC_CACHE = pathlib.Path(__file__).resolve().parent.parent / "out" / "pipeline_spec.json"


def wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def ideal_port_heading(k):
    """Design heading after k gates; the compiled spec stores the achieved ones."""
    return wrap_deg(180.0 - BEND * k)


def _c_from_frame(E, h_in_deg, t_k, d):
    """Control ball state: anti-parallel to lane heading h_in, starboard
    offset d, speed C_SPEED, arriving at aim point E at time t_k."""
    h = np.radians(h_in_deg)
    u = np.array([np.cos(h), np.sin(h)])
    n = np.array([u[1], -u[0]])
    pos = np.asarray(E) + u * (C_SPEED * t_k) + n * d
    return {"c_pos": pos.tolist(), "c_vel": (-u * C_SPEED).tolist()}


def _lane_normal(h_in_deg):
    h = np.radians(h_in_deg)
    return np.array([np.sin(h), -np.cos(h)])  # starboard of heading h


def _bodies(gates, bits, perturb_b=0.0):
    b = dict(BALL_B)
    b["pos"] = [b["pos"][0], b["pos"][1] + perturb_b]
    bodies = [b]
    for g, bit in zip(gates, bits):
        if bit:
            bodies.append({"m": 1.0, "pos": list(g["c_pos"]),
                           "vel": list(g["c_vel"])})
    return bodies


def _run_raw(gates, bits, t_end, n_samples, perturb_b=0.0, rtol=1e-12):
    res = nbody.integrate(_bodies(gates, bits, perturb_b), t_end,
                          n_samples=n_samples, rtol=rtol, atol=rtol)
    res["b_index"] = 0
    return res


def _heading(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _b_state(res, t_q):
    idx = min(np.searchsorted(res["t"], t_q), len(res["t"]) - 1)
    return res["traj"][0, idx], res["vel"][0, idx]


# ---------------------------------------------------------------------------
# Greedy seed: one cheap forward pass to fix each gate's lane frame and a
# starting offset. No relaxation, no shooting — the global solve does that.
# ---------------------------------------------------------------------------

def _best_effort_d(f, lo=0.4, hi=2.4, step=0.15):
    """Offset giving the smallest |bend error|; brackets a root if one exists.
    Tolerant by design — this only seeds the global solve."""
    def safe(d):
        try:
            return f(d)
        except RuntimeError:      # near-collision blows the integrator step
            return np.nan
    grid = np.arange(lo, hi + 1e-9, step)
    vals = [safe(d) for d in grid]
    for a, b, fa, fb in zip(grid, grid[1:], vals, vals[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            return float(brentq(safe, a, b, xtol=1e-3))
    finite = [(abs(v), d) for v, d in zip(vals, grid) if np.isfinite(v)]
    return float(min(finite)[1]) if finite else float(grid[len(grid) // 2])


def _seed_gate(committed, t_k, n_samples):
    """Lane frame + starting offset for the gate at T_k, given the controls
    committed so far (all present)."""
    bits = (1,) * len(committed)
    r0 = _run_raw(committed, bits, t_k, n_samples, rtol=SOLVE_RTOL)
    E, v0 = _b_state(r0, t_k)
    E, h_in = E.tolist(), _heading(v0)

    def bend_err(dd):
        cand = committed + [_c_from_frame(E, h_in, t_k, dd)]
        r = _run_raw(cand, bits + (1,), t_k + LANE_DT + 1.0, n_samples,
                     rtol=SOLVE_RTOL)
        _, v_pre = _b_state(r, t_k - 4.0)
        _, v_post = _b_state(r, t_k + LANE_DT + 1.0)
        return wrap_deg(_heading(v_post) - _heading(v_pre) + BEND)

    d = _best_effort_d(bend_err)
    gate = _c_from_frame(E, h_in, t_k, d)
    u = np.array(gate["c_vel"]) / np.linalg.norm(gate["c_vel"])  # flight dir
    gate.update({"t": t_k, "d": d, "E": E, "h_in": h_in,
                 "c_pos0": gate["c_pos"], "u": u.tolist(),
                 "n": _lane_normal(h_in).tolist()})
    return gate


def _greedy_seed(n_gates, n_samples):
    gates = []
    for k in range(1, n_gates + 1):
        gates.append(_seed_gate(list(gates), DT * k, n_samples))
    return gates


# ---------------------------------------------------------------------------
# Global solve: two knobs per control — a slide along its flight line (timing)
# and a slide along its lane normal (impact parameter) — solved together so
# the all-ones run lands every port with every control on time.
# ---------------------------------------------------------------------------

W_TIME = 8.0  # weight on timing residual (time units -> ~deg) to balance LM


def _controls_from_x(gates, x):
    controls = []
    for i, g in enumerate(gates):
        a, p = x[2 * i], x[2 * i + 1]
        c_pos = (np.array(g["c_pos0"]) + a * np.array(g["u"])
                 + p * np.array(g["n"])).tolist()
        controls.append({"c_pos": c_pos, "c_vel": g["c_vel"], "t": g["t"]})
    return controls


def _t_closest(res, ci, t_k, half=8.0):
    """Sub-sample time of closest B-C_ci approach in [t_k-half, t_k+half]."""
    t = res["t"]
    w = np.where((t >= t_k - half) & (t <= t_k + half))[0]
    d = np.linalg.norm(res["traj"][0, w] - res["traj"][ci, w], axis=1)
    j = int(np.argmin(d))
    if 0 < j < len(w) - 1:  # parabolic refine around the sampled minimum
        y0, y1, y2 = d[j - 1], d[j], d[j + 1]
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        dt = t[w[1]] - t[w[0]]
        return float(t[w[j]] + shift * dt)
    return float(t[w[j]])


def _solve_residuals(x, gates, t_end, n_samples):
    controls = _controls_from_x(gates, x)
    n = len(gates)
    try:
        res = _run_raw(controls, (1,) * n, t_end, n_samples, rtol=SOLVE_RTOL)
    except RuntimeError:
        return np.full(2 * n, 180.0)  # collision: push the solver away
    out = []
    for k, g in enumerate(gates, 1):
        # LOCAL bend of THIS gate only (heading after minus heading before),
        # pinned to -BEND. Measuring the cumulative port heading instead lets
        # adjacent gates trade error and settle into a degenerate split where
        # each turns half a port (the exact 27 deg = BEND/2 wall); the local
        # target forbids that — every gate must make its own full turn.
        _, v_pre = _b_state(res, g["t"] - 4.0)
        _, v_post = _b_state(res, g["t"] + LANE_DT)
        bend = wrap_deg(_heading(v_post) - _heading(v_pre))
        out.append(wrap_deg(bend + BEND))                           # bend
        out.append(W_TIME * (_t_closest(res, k, g["t"]) - g["t"]))  # timing
    return np.array(out)


def _measure_ports(gates, t_end, n_samples):
    """Achieved port headings AND each gate's realized impact parameter (the
    launch-frame offset d is not the impact parameter once the control is
    deflected in flight, so we measure the true closest approach)."""
    res = _run_raw(gates, (1,) * len(gates), t_end, n_samples)
    ports = [180.0]  # port 0: the untouched straight lane
    impact = []
    for k, g in enumerate(gates, 1):
        _, v = _b_state(res, g["t"] + LANE_DT)
        ports.append(_heading(v))
        w = np.where((res["t"] >= g["t"] - 8) & (res["t"] <= g["t"] + 8))[0]
        impact.append(float(np.min(np.linalg.norm(
            res["traj"][0, w] - res["traj"][k, w], axis=1))))
    return ports, impact


def _cache_key(n_gates):
    """Every constant that shapes the compiled machine belongs in the cache
    key — a spec compiled under old values must not be silently reused."""
    return {"n_gates": n_gates, "bend": BEND, "dt": DT, "c_speed": C_SPEED,
            "tail": TAIL, "lane_dt": LANE_DT, "w_time": W_TIME,
            "solve_rtol": SOLVE_RTOL}


def compile_pipeline(n_gates=4, use_cache=True, n_samples=250, verbose=False):
    """Seed greedily, then solve all controls together (timing + impact
    parameter per gate) with damped least squares. The spec records the
    achieved port headings — the machine defines its own ports."""
    key = _cache_key(n_gates)
    if use_cache and SPEC_CACHE.exists():
        spec = json.loads(SPEC_CACHE.read_text())
        if all(spec.get(k) == v for k, v in key.items()):
            return spec

    log = print if verbose else (lambda *a: None)
    t_end = DT * n_gates + TAIL
    gates = _greedy_seed(n_gates, n_samples)
    log(f"  seed done: d={[round(g['d'], 2) for g in gates]}")

    evals = [0]

    def resid(x):
        evals[0] += 1
        r = _solve_residuals(x, gates, t_end, n_samples)
        if verbose and evals[0] % 20 == 0:
            log(f"  eval {evals[0]}: worst bend {np.max(np.abs(r[0::2])):.1f} deg")
        return r

    # diff_step must be large enough that the FD Jacobian still "sees" a
    # control that is currently missing B by a unit or two — a 1e-3 step
    # registers no bend change for a far control (flat column, dead gate),
    # so the solver never pulls it back in. 0.05 units of launch shift moves
    # the impact parameter enough to register even from a near-miss.
    sol = least_squares(
        resid, np.zeros(2 * n_gates), method="trf", x_scale="jac",
        diff_step=0.05, xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=250)

    bend_res = sol.fun[0::2]
    worst = float(np.max(np.abs(bend_res)))
    log(f"  solve done: {sol.nfev} evals, worst bend {worst:.2f} deg")
    if worst > 3.0:
        raise RuntimeError(
            f"global solve did not converge: worst port error {worst:.1f} deg, "
            f"bend residuals={np.round(bend_res, 2)}")

    for i, g in enumerate(gates):
        a, p = sol.x[2 * i], sol.x[2 * i + 1]
        g["c_pos"] = (np.array(g["c_pos0"]) + a * np.array(g["u"])
                      + p * np.array(g["n"])).tolist()
        g["d"] = float(g["d"] + p)

    ports, impact = _measure_ports(gates, t_end, n_samples=1200)
    gaps = [abs(wrap_deg(a - b)) for a, b in zip(ports, ports[1:])]
    if min(gaps) < 35.0:
        raise RuntimeError(f"compiled ports not separated: {np.round(ports, 1)}")

    spec = {**key, "t_end": t_end,
            "gates": [{"t": g["t"], "d": g["d"], "b": b, "c_pos": g["c_pos"],
                       "c_vel": g["c_vel"]} for g, b in zip(gates, impact)],
            "ports": ports,
            "solve": {"worst_deg": worst, "nfev": int(sol.nfev)}}
    SPEC_CACHE.parent.mkdir(exist_ok=True)
    SPEC_CACHE.write_text(json.dumps(spec))
    return spec


def run_pipeline(spec, bits, n_samples=600, perturb_b=0.0):
    assert len(bits) == spec["n_gates"]
    return _run_raw(spec["gates"], bits, spec["t_end"], n_samples,
                    perturb_b=perturb_b)


def classify_pipeline(spec, res):
    """(gates passed, heading margin in deg). Port k = first zero at gate k+1.
    Ports are the compiled machine's achieved lane headings."""
    _, v = _b_state(res, spec["t_end"])
    heading = _heading(v)
    errs = [abs(wrap_deg(heading - p)) for p in spec["ports"]]
    k = int(np.argmin(errs))
    return k, float(errs[k])
