"""Physics invariants of the integrator.

Every logic claim in this project rests on the integrator being trustworthy:
symplectic-grade energy behavior, exact momentum bookkeeping, time-reversal
symmetry, and agreement with the closed-form two-body scattering solution.
"""

import numpy as np
import pytest

from slingshot import nbody

TWO_BODY = [
    {"m": 1.0, "pos": [-10.0, 0.5], "vel": [1.0, 0.0]},
    {"m": 1.0, "pos": [10.0, -0.5], "vel": [-1.0, 0.0]},
]

THREE_BODY = [
    {"m": 1.0, "pos": [-10.0, 0.5], "vel": [1.0, 0.0]},
    {"m": 1.0, "pos": [10.0, -0.5], "vel": [-1.0, 0.0]},
    {"m": 2.0, "pos": [3.0, 9.0], "vel": [-0.3, -0.4]},
]


def total_momentum(res):
    n = len(res["masses"])
    return (res["masses"][:, None] * res["vel"][:, -1, :]).sum(axis=0)


class TestConservation:
    def test_energy_two_body(self):
        res = nbody.integrate([dict(b) for b in TWO_BODY], 22.0)
        assert res["energy_drift"] < 1e-10

    def test_energy_three_body(self):
        res = nbody.integrate([dict(b) for b in THREE_BODY], 40.0)
        assert res["energy_drift"] < 1e-9

    def test_momentum_three_body(self):
        bodies = [dict(b) for b in THREE_BODY]
        p0 = sum(np.array(b["vel"]) * b["m"] for b in bodies)
        res = nbody.integrate(bodies, 40.0)
        assert np.allclose(total_momentum(res), p0, atol=1e-9)

    def test_com_drifts_linearly(self):
        bodies = [dict(b) for b in THREE_BODY]
        res = nbody.integrate(bodies, 40.0)
        m = res["masses"]
        com = (m[:, None, None] * res["traj"]).sum(axis=0) / m.sum()
        v_com = (m[:, None] * res["vel"][:, 0, :]).sum(axis=0) / m.sum()
        expect = com[0] + np.outer(res["t"], v_com)
        assert np.allclose(com, expect, atol=1e-9)


class TestTimeReversal:
    def test_flyby_retraces_itself(self):
        """Integrate through the gate-1 flyby, flip velocities, integrate the
        same span again: bodies must land exactly on their launch points."""
        res = nbody.integrate([dict(b) for b in TWO_BODY], 22.0)
        back = [
            {"m": m, "pos": res["traj"][i, -1].tolist(),
             "vel": (-res["vel"][i, -1]).tolist()}
            for i, m in enumerate(res["masses"])
        ]
        res2 = nbody.integrate(back, 22.0)
        for i, b in enumerate(TWO_BODY):
            assert np.allclose(res2["traj"][i, -1], b["pos"], atol=1e-6)


class TestKeplerScattering:
    def test_scattering_angle_matches_closed_form(self):
        """b=1, v_rel=2, G(m1+m2)=2 -> theta = 2*atan(0.5) = 53.13 deg.
        Launched from +-100 so the finite-range correction is tiny."""
        bodies = [
            {"m": 1.0, "pos": [-100.0, 0.5], "vel": [1.0, 0.0]},
            {"m": 1.0, "pos": [100.0, -0.5], "vel": [-1.0, 0.0]},
        ]
        res = nbody.integrate(bodies, 200.0, n_samples=400)
        v = res["vel"][1, -1]
        theta = np.degrees(np.arctan2(v[1], -v[0]))
        assert theta == pytest.approx(np.degrees(2 * np.arctan(0.5)), abs=0.5)

    def test_closest_approach_matches_closed_form(self):
        """E=2, L=2 per reduced mass -> r_min = (sqrt(5)-1)/2."""
        bodies = [
            {"m": 1.0, "pos": [-100.0, 0.5], "vel": [1.0, 0.0]},
            {"m": 1.0, "pos": [100.0, -0.5], "vel": [-1.0, 0.0]},
        ]
        res = nbody.integrate(bodies, 200.0, n_samples=8000)
        d = np.linalg.norm(res["traj"][0] - res["traj"][1], axis=1)
        assert d.min() == pytest.approx((np.sqrt(5) - 1) / 2, abs=2e-3)


class TestDeterminism:
    def test_bitwise_repeatable(self):
        r1 = nbody.integrate([dict(b) for b in THREE_BODY], 30.0)
        r2 = nbody.integrate([dict(b) for b in THREE_BODY], 30.0)
        assert np.array_equal(r1["traj"], r2["traj"])
