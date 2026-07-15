"""Contract for the gravitational circuit compiler.

The pipeline is a chain of N flyby gates: ball B always flies; bit k is the
presence of control ball C_k, aimed at B's post-gate-(k-1) trajectory and
calibrated so a hit bends B's heading by one port step (-54 deg). B's final
heading therefore encodes THE INDEX OF THE FIRST 0 BIT — a priority encoder —
and "all gates passed" is an N-input AND. The flagship is N=4: one 5-body
machine, 16 input cases, five ports.

Depth is capped near four gates by chaotic control-flight (a 5th control
can't be aimed through the accumulated field) — that ceiling is a documented
result, not a defect, so the contract targets N=4.
"""

import itertools

import numpy as np
import pytest

from slingshot import pipeline

N_GATES = 4


@pytest.fixture(scope="session")
def spec():
    return pipeline.compile_pipeline(n_gates=N_GATES)


def first_zero(bits):
    return bits.index(0) if 0 in bits else len(bits)


class TestCompiler:
    def test_spec_shape(self, spec):
        assert spec["n_gates"] == N_GATES
        assert len(spec["gates"]) == N_GATES

    def test_gates_hit_at_sane_impact_parameter(self, spec):
        """Every control must actually encounter B — the realized closest
        approach (not the launch-frame offset) sits in a real gate range."""
        for g in spec["gates"]:
            assert 0.2 < g["b"] < 3.0, f"gate at t={g['t']} impact {g['b']:.2f}"

    def test_gates_are_time_ordered_and_spaced(self, spec):
        times = [g["t"] for g in spec["gates"]]
        assert times == sorted(times)
        assert min(np.diff(times)) >= 8.0

    def test_ports_near_design_and_separated(self, spec):
        """The machine defines its own ports; they must still track the design
        fan (within 10 deg of 180 - 54k) and stay >= 35 deg apart."""
        ports = spec["ports"]
        assert len(ports) == N_GATES + 1
        for k, p in enumerate(ports):
            err = pipeline.wrap_deg(p - pipeline.ideal_port_heading(k))
            assert abs(err) < 10.0, f"port {k} off design by {err:.1f} deg"
        gaps = [abs(pipeline.wrap_deg(a - b)) for a, b in zip(ports, ports[1:])]
        assert min(gaps) >= 35.0

    def test_lanes_are_settled_between_gates(self, spec):
        """All-ones run: B's heading must be stable (not mid-turn) on every
        inter-gate lane — the gate is an event, the lane is a wire."""
        res = pipeline.run_pipeline(spec, (1,) * N_GATES, n_samples=1500)
        for k in range(1, N_GATES + 1):
            hs = []
            for dt_check in (5.0, 9.0):
                idx = np.searchsorted(res["t"], spec["gates"][k - 1]["t"] + dt_check)
                idx = min(idx, len(res["t"]) - 1)
                v = res["vel"][res["b_index"], idx]
                hs.append(np.degrees(np.arctan2(v[1], v[0])))
            drift = abs(pipeline.wrap_deg(hs[1] - hs[0]))
            assert drift < 2.5, f"lane after gate {k} still turning: {drift:.2f} deg"


class TestPriorityEncoder:
    @pytest.mark.parametrize("bits", list(itertools.product((0, 1), repeat=N_GATES)),
                             ids=lambda b: "".join(map(str, b)))
    def test_all_cases(self, spec, bits):
        res = pipeline.run_pipeline(spec, bits)
        k, margin = pipeline.classify_pipeline(spec, res)
        assert k == first_zero(list(bits)), f"bits={bits}: routed to port {k}"
        # decision boundary is BEND/2 = 27 deg; demand a real cushion
        assert margin < 20.0, f"bits={bits}: heading margin {margin:.1f} deg"

    def test_and_is_the_last_port(self, spec):
        res = pipeline.run_pipeline(spec, (1,) * N_GATES)
        k, _ = pipeline.classify_pipeline(spec, res)
        assert k == N_GATES

    def test_first_zero_selects_port(self, spec):
        """Spot-check the priority-encoder semantics on a mixed input."""
        res = pipeline.run_pipeline(spec, (1, 1, 0, 1))
        k, _ = pipeline.classify_pipeline(spec, res)
        assert k == 2

    def test_energy_drift_deep_run(self, spec):
        res = pipeline.run_pipeline(spec, (1,) * N_GATES)
        assert res["energy_drift"] < 1e-9


class TestChaosVsDepth:
    def test_error_multiplies_across_the_chain(self, spec):
        delta = 1e-9
        nom = pipeline.run_pipeline(spec, (1,) * N_GATES, n_samples=2000)
        pert = pipeline.run_pipeline(spec, (1,) * N_GATES, n_samples=2000,
                                     perturb_b=delta)
        bi = nom["b_index"]
        sep = np.linalg.norm(nom["traj"][bi] - pert["traj"][bi], axis=1)
        t = nom["t"]
        seps = [float(np.interp(spec["gates"][k]["t"] + 4.0, t, sep))
                for k in range(N_GATES)]
        # ballistic before gate 1, then growth through the chain
        assert seps[0] > delta
        assert seps[-1] > 5 * seps[0]
        # ~1 digit per gate -> not catastrophic over four gates
        assert seps[-1] / delta < 1e7

    def test_digits_per_gate_within_float64_budget(self, spec):
        delta = 1e-9
        nom = pipeline.run_pipeline(spec, (1,) * N_GATES, n_samples=1200)
        pert = pipeline.run_pipeline(spec, (1,) * N_GATES, n_samples=1200,
                                     perturb_b=delta)
        bi = nom["b_index"]
        sep = np.linalg.norm(nom["traj"][bi] - pert["traj"][bi], axis=1)
        digits_per_gate = np.log10(sep[-1] / delta) / N_GATES
        assert digits_per_gate < 1.5
