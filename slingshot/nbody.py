"""Planar Newtonian n-body integrator for slingshot logic gates.

Units: G = 1. State vector layout: [x1, y1, ..., xN, yN, vx1, vy1, ..., vxN, vyN].
Point masses, no softening — sharp scattering is the whole point, so gate
geometries must be designed to avoid actual collisions.
"""

import numpy as np
from scipy.integrate import solve_ivp

G = 1.0


def rhs(t, y, masses):
    n = len(masses)
    pos = y[: 2 * n].reshape(n, 2)
    vel = y[2 * n :].reshape(n, 2)
    acc = np.zeros_like(pos)
    for i in range(n):
        dr = pos - pos[i]
        r2 = np.einsum("ij,ij->i", dr, dr)
        r2[i] = np.inf
        acc[i] = np.sum((G * masses / (r2 * np.sqrt(r2)))[:, None] * dr, axis=0)
    return np.concatenate([vel.ravel(), acc.ravel()])


def total_energy(y, masses):
    n = len(masses)
    pos = y[: 2 * n].reshape(n, 2)
    vel = y[2 * n :].reshape(n, 2)
    kinetic = 0.5 * np.sum(masses * np.einsum("ij,ij->i", vel, vel))
    potential = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            potential -= G * masses[i] * masses[j] / np.linalg.norm(pos[i] - pos[j])
    return kinetic + potential


def integrate(bodies, t_end, n_samples=600, rtol=1e-12, atol=1e-12):
    """Integrate a list of bodies to t_end.

    bodies: list of dicts with keys 'm', 'pos', 'vel'.
    Returns dict with times (n_samples,), trajectories (N, n_samples, 2),
    velocities (N, n_samples, 2), and relative energy drift.
    """
    masses = np.array([b["m"] for b in bodies])
    y0 = np.concatenate(
        [np.array([b["pos"] for b in bodies]).ravel(),
         np.array([b["vel"] for b in bodies]).ravel()]
    )
    times = np.linspace(0.0, t_end, n_samples)
    sol = solve_ivp(
        rhs, (0.0, t_end), y0, args=(masses,), method="DOP853",
        t_eval=times, rtol=rtol, atol=atol, dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    n = len(masses)
    traj = sol.y[: 2 * n].T.reshape(-1, n, 2).transpose(1, 0, 2)
    vels = sol.y[2 * n :].T.reshape(-1, n, 2).transpose(1, 0, 2)
    e0 = total_energy(y0, masses)
    e1 = total_energy(sol.y[:, -1], masses)
    drift = abs((e1 - e0) / e0) if e0 != 0 else abs(e1 - e0)
    return {"t": times, "traj": traj, "vel": vels, "energy_drift": drift, "masses": masses}
