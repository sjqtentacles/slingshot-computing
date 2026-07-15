"""The SWITCH primitive: truth table, analytic agreement, digital robustness."""

import numpy as np
import pytest

from slingshot import gates


class TestSwitchTruthTable:
    @pytest.mark.parametrize("a_present,expected_port", [(False, 0), (True, 1)])
    def test_ports(self, a_present, expected_port):
        res = gates.run_switch(a_present)
        port, _ = gates.classify_port(res)
        assert port == expected_port

    def test_energy_drift(self):
        res = gates.run_switch(True)
        assert res["energy_drift"] < 1e-10


class TestSwitchPhysics:
    def test_deflection_near_kepler_angle(self):
        """Finite-range launch adds ~1 deg to the asymptotic 53.13 deg."""
        res = gates.run_switch(True)
        _, angle = gates.classify_port(res)
        assert angle == pytest.approx(53.13, abs=2.0)

    def test_both_balls_deflect_symmetrically(self):
        """Equal masses, zero net momentum: A's exit mirrors B's."""
        res = gates.run_switch(True)
        va, vb = res["vel"][0, -1], res["vel"][1, -1]
        assert np.allclose(va, -vb, atol=1e-9)


class TestDigitalRobustness:
    """A logic gate must tolerate small analog error: the port decision may
    not flip under launch perturbations far larger than integration error."""

    @pytest.mark.parametrize("dy", [1e-3, -1e-3, 5e-3])
    def test_port_stable_under_launch_jitter(self, dy):
        for a_present, expected in ((True, 1), (False, 0)):
            bodies_port = []
            from slingshot.gates import BALL_A, BALL_B
            from slingshot import nbody
            b = dict(BALL_B)
            b["pos"] = [b["pos"][0], b["pos"][1] + dy]
            bodies = ([dict(BALL_A), b] if a_present else [b])
            res = nbody.integrate(bodies, gates.T_END)
            res["b_index"] = 1 if a_present else 0
            port, _ = gates.classify_port(res)
            assert port == expected
