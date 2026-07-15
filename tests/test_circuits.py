"""Half adder and two-gate cascade: full truth tables, margins, calibration."""

import numpy as np
import pytest

from slingshot import circuits


class TestHalfAdder:
    @pytest.mark.parametrize("a,b", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_truth_table(self, a, b):
        res = circuits.run_half_adder(a, b)
        s, carry = circuits.classify_half_adder(res)
        assert (s, carry) == ((a + b) % 2, (a + b) // 2)

    def test_empty_case_is_silent(self):
        res = circuits.run_half_adder(0, 0)
        assert circuits.classify_half_adder(res) == (0, 0)
        assert len(res["names"]) == 0


class TestCascade:
    @pytest.mark.parametrize("a,c", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_truth_table(self, a, c):
        res = circuits.run_cascade(a, c)
        port = circuits.classify_cascade(res)
        assert {"straight": 0, "bend1": 0, "bend2": 1}[port] == (a & c)

    def test_energy_drift_all_cases(self):
        for a in (0, 1):
            for c in (0, 1):
                assert circuits.run_cascade(a, c)["energy_drift"] < 1e-10

    def test_calibration_hits_target_angle(self):
        """The calibrated wire must bring the double-bent B back to within
        half a degree of anti-parallel exit."""
        res = circuits.run_cascade(1, 1)
        assert abs(circuits._exit_angle(res)) < 0.5

    def test_crosstalk_stays_below_margin(self):
        """A=0, C=1: C's lane must not lift the straight ball anywhere near
        the bend2 decision threshold (y=4). This is the no-insulation bound."""
        res = circuits.run_cascade(0, 1)
        y = res["traj"][res["b_index"], -1, 1]
        assert y < 2.0

    def test_output_separation_margin(self):
        """Final B positions of the three logic routes are far apart compared
        to every perturbation in play."""
        ends = {}
        for a, c in ((0, 0), (1, 0), (1, 1)):
            r = circuits.run_cascade(a, c)
            ends[(a, c)] = r["traj"][r["b_index"], -1]
        pairs = [((0, 0), (1, 0)), ((0, 0), (1, 1)), ((1, 0), (1, 1))]
        for k1, k2 in pairs:
            assert np.linalg.norm(ends[k1] - ends[k2]) > 5.0


class TestChaosTax:
    def test_error_grows_through_each_gate(self):
        delta = 1e-8
        nom = circuits.run_cascade(1, 1, n_samples=800)
        pert = circuits.run_cascade(1, 1, n_samples=800, perturb_b=delta)
        bi = nom["b_index"]
        sep = np.linalg.norm(nom["traj"][bi] - pert["traj"][bi], axis=1)
        t = nom["t"]
        s_pre = np.interp(8.0, t, sep)
        s_mid = np.interp(16.0, t, sep)
        s_end = sep[-1]
        assert s_pre == pytest.approx(delta, rel=0.5)   # ballistic: no growth yet
        assert s_mid > 3 * delta                        # gate 1 multiplied it
        assert s_end > 3 * s_mid                        # gate 2 multiplied again
