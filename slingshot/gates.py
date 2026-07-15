"""Gate geometries.

The SWITCH gate is the primitive (same role as the collision in the
Fredkin-Toffoli billiard-ball computer): bit A is the presence/absence of
ball A. Ball B always flies. If A is absent, B exits straight through
port 0. If A is present, a close hyperbolic flyby deflects B into port 1.

Geometry (G = 1, equal unit masses, zero total momentum so lab = CM frame):
  A starts at (-10, +0.5) moving +x at speed 1
  B starts at (+10, -0.5) moving -x at speed 1
Impact parameter b = 1, relative speed v = 2, so the Kepler scattering
angle is tan(theta/2) = G(mA+mB)/(b v^2) = 0.5  ->  theta ~ 53 deg.
Closest approach ~0.62, comfortably away from singularity.
"""

from . import nbody

BALL_A = {"m": 1.0, "pos": [-10.0, 0.5], "vel": [1.0, 0.0]}
BALL_B = {"m": 1.0, "pos": [10.0, -0.5], "vel": [-1.0, 0.0]}

T_END = 22.0

# Output classification: B's exit direction, degrees CCW from -x axis.
PORT_SPLIT_DEG = 25.0  # below -> port 0 (straight), above -> port 1 (deflected)


def run_switch(a_present, n_samples=600):
    bodies = ([dict(BALL_A), dict(BALL_B)] if a_present else [dict(BALL_B)])
    result = nbody.integrate(bodies, T_END, n_samples=n_samples)
    result["b_index"] = 1 if a_present else 0
    result["a_present"] = a_present
    return result


def classify_port(result):
    import numpy as np

    v = result["vel"][result["b_index"], -1]
    # angle of exit velocity measured CCW from the -x axis
    angle = np.degrees(np.arctan2(v[1], -v[0]))
    return (1 if angle > PORT_SPLIT_DEG else 0), angle
