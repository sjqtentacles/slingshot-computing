"""Composed circuits built on the SWITCH primitive.

Half adder: the flyby is Fredkin-Toffoli's "interaction gate". With BOTH
balls as input bits, the deflected exits carry A AND B (the carry bit) and
the straight-through exits carry A XOR B at the detector level (exactly one
straight exit is occupied iff exactly one input was 1). So one scattering
event + four detector regions = a half adder.

Cascade: the deflected output of gate 1 becomes the input signal of gate 2.
Ball B always flies; bit A gates the first flyby, bit C is a third ball
placed on a collision course with B's *post-gate-1* trajectory (anti-parallel
closing, gate-1-like geometry rotated onto the new lane). The LAUNCH offset
is not the impact parameter: C's long-range pull drags B off course during
the whole approach, so the offset is calibrated on the full simulation
(calibrated_d) until the ACHIEVED closest approach is a proper gate-1-scale
encounter. B ends double-bent iff A AND C. Composition is what makes
ballistic logic universal; it is also where chaos starts charging per-gate
interest.
"""

import numpy as np

from . import nbody
from .gates import BALL_A, BALL_B, PORT_SPLIT_DEG, T_END as GATE1_T_END

# Gate 2 must sit well downstream of gate 1: deflection falls off only as
# ~1/impact-parameter (no shielding, no insulation), so the straight lane and
# the bent lane need to diverge ~9+ units before C's lane can cross without
# bending an innocent straight-through B. At T2=20 the lanes are ~9 apart.
T2 = 20.0      # scheduled time of the second flyby
T_END = 31.0   # cascade integration horizon

_frame_cache = None
_cal_d_cache = None


def _gate1_frame():
    """B's position/heading at T2 from a C-free run of gate 1."""
    global _frame_cache
    if _frame_cache is None:
        res = nbody.integrate([dict(BALL_A), dict(BALL_B)], T2, n_samples=400)
        pos, vel = res["traj"][1][-1], res["vel"][1][-1]
        u = vel / np.linalg.norm(vel)
        _frame_cache = (pos, u, np.array([-u[1], u[0]]))
    return _frame_cache


def ball_c(d):
    """Third ball, aimed at B's post-gate-1 trajectory.

    Launched anti-parallel to B's T2 heading, offset d to B's port side,
    timed to arrive at T2. The naive billiard aim (d = 1 impact parameter)
    fails: gravity has no shielding, so C's long-range pull drags B off its
    gate-1 course during the entire approach and the effective impact
    parameter collapses. d must be calibrated in situ — see calibrated_d().
    """
    pos, u, n = _gate1_frame()
    return {"m": 1.0, "pos": (pos + u * T2 + n * d).tolist(),
            "vel": (-u).tolist()}


def _exit_angle(res):
    v = res["vel"][res["b_index"], -1]
    return float(np.degrees(np.arctan2(v[1], -v[0])))


def calibrated_d():
    """Aim offset making the double-bent B exit anti-parallel (0 deg from -x),
    the mirror of gate 1. Root-found on the full 3-body simulation."""
    global _cal_d_cache
    if _cal_d_cache is None:
        from scipy.optimize import brentq
        f = lambda d: _exit_angle(run_cascade(1, 1, n_samples=200, d=d))
        grid = np.arange(1.0, 3.51, 0.5)
        vals = [f(d) for d in grid]
        for lo, hi, flo, fhi in zip(grid, grid[1:], vals, vals[1:]):
            if flo * fhi < 0:
                _cal_d_cache = brentq(f, lo, hi, xtol=1e-4)
                break
        else:
            raise RuntimeError(f"no sign change on calibration grid: {vals}")
    return _cal_d_cache


def run_cascade(a, c, n_samples=700, perturb_b=0.0, d=None):
    bodies, names = [], []
    if a:
        bodies.append(dict(BALL_A)); names.append("A")
    b = dict(BALL_B)
    b["pos"] = [b["pos"][0], b["pos"][1] + perturb_b]
    bodies.append(b); names.append("B")
    if c:
        bodies.append(ball_c(d if d is not None else calibrated_d()))
        names.append("C")
    res = nbody.integrate(bodies, T_END, n_samples=n_samples)
    res["names"] = names
    res["b_index"] = names.index("B")
    return res


def classify_cascade(res):
    """B's route: straight (out 0), one bend (out 0), two bends (out 1).

    The calibrated double bend exits anti-parallel (angle ~0) but displaced
    upward from the straight lane, so angle splits off bend1 and height
    splits bend2 from straight."""
    angle = _exit_angle(res)
    p = res["traj"][res["b_index"], -1]
    if angle > PORT_SPLIT_DEG:
        return "bend1"
    return "bend2" if p[1] > 4.0 else "straight"


def run_half_adder(a, b, n_samples=600):
    bodies, names = [], []
    if a:
        bodies.append(dict(BALL_A)); names.append("A")
    if b:
        bodies.append(dict(BALL_B)); names.append("B")
    if not bodies:
        return {"names": [], "traj": np.zeros((0, n_samples, 2)),
                "vel": np.zeros((0, n_samples, 2)),
                "t": np.linspace(0, GATE1_T_END, n_samples),
                "energy_drift": 0.0, "masses": np.array([])}
    res = nbody.integrate(bodies, GATE1_T_END, n_samples=n_samples)
    res["names"] = names
    return res


# detector regions for the half adder, (center, radius): a ball finishing
# inside SUM regions raises the sum bit, inside CARRY regions the carry bit
SUM_REGIONS = [((12.0, 0.5), 1.5), ((-12.0, -0.5), 1.5)]
CARRY_REGIONS = [((-8.3, 10.1), 2.0), ((8.3, -10.1), 2.0)]


def classify_half_adder(res):
    finals = [res["traj"][i, -1] for i in range(len(res["names"]))]

    def occupied(regions):
        return any(np.hypot(p[0] - cx, p[1] - cy) < r
                   for p in finals for (cx, cy), r in regions)

    return int(occupied(SUM_REGIONS)), int(occupied(CARRY_REGIONS))
