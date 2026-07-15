"""Error paths and guard rails — the branches a healthy run never visits.

These exist so that failure modes fail LOUDLY and correctly: integrator
step-size underflow on collisions, solver fallbacks, stale-cache detection,
and input validation.
"""

import json

import numpy as np
import pytest

from slingshot import circuits, nbody, pipeline


class TestIntegratorFailure:
    def test_head_on_collision_raises(self):
        """Two point masses on an exact head-on course drive the adaptive
        step size to underflow — integrate() must surface that as an error,
        not return garbage."""
        bodies = [{"m": 1.0, "pos": [-1.0, 0.0], "vel": [1.0, 0.0]},
                  {"m": 1.0, "pos": [1.0, 0.0], "vel": [-1.0, 0.0]}]
        with pytest.raises(RuntimeError, match="integration failed"):
            nbody.integrate(bodies, 5.0, n_samples=50)


class TestPipelineGuards:
    def test_run_pipeline_rejects_wrong_bit_count(self):
        spec = pipeline.compile_pipeline()
        with pytest.raises(AssertionError):
            pipeline.run_pipeline(spec, (1, 1))  # needs n_gates bits

    def test_best_effort_d_with_no_root_returns_min_abs(self):
        d = pipeline._best_effort_d(lambda x: 5.0 + x, lo=0.5, hi=1.0, step=0.25)
        assert d == pytest.approx(0.5)  # smallest |err| on the grid

    def test_best_effort_d_survives_total_collision(self):
        """Every probe raising (collision) must still return a grid point,
        never crash the seed pass."""
        def boom(_):
            raise RuntimeError("collision")
        d = pipeline._best_effort_d(boom, lo=0.5, hi=1.0, step=0.25)
        assert 0.5 <= d <= 1.0

    def test_solve_residuals_collision_returns_penalty(self):
        """A collision inside a solver probe becomes a large flat residual
        (pushes the trust region away), not an exception."""
        # a control aimed exactly head-on at the signal ball's launch lane
        gates = [{"c_pos0": [-10.0, -0.5], "u": [1.0, 0.0], "n": [0.0, 1.0],
                  "c_vel": [1.0, 0.0], "t": 12.0, "d": 0.0}]
        r = pipeline._solve_residuals(np.zeros(2), gates, 24.0, 100)
        assert r.shape == (2,)
        assert np.all(r == 180.0)


class TestSpecCache:
    def _fake_spec(self, n_gates=4):
        key = pipeline._cache_key(n_gates)
        return {**key, "t_end": 1.0, "gates": [], "ports": [180.0],
                "solve": {"worst_deg": 0.0, "nfev": 0}}

    def test_cache_hit_returns_stored_spec(self, tmp_path, monkeypatch):
        cache = tmp_path / "spec.json"
        spec = self._fake_spec()
        cache.write_text(json.dumps(spec))
        monkeypatch.setattr(pipeline, "SPEC_CACHE", cache)
        assert pipeline.compile_pipeline(use_cache=True) == spec

    def test_cache_missed_when_any_constant_changes(self, tmp_path, monkeypatch):
        """A spec compiled under different TAIL/LANE_DT/W_TIME/SOLVE_RTOL
        must NOT be reused — the audit found these were silently ignored."""
        cache = tmp_path / "spec.json"
        cache.write_text(json.dumps(self._fake_spec()))
        monkeypatch.setattr(pipeline, "SPEC_CACHE", cache)
        monkeypatch.setattr(pipeline, "TAIL", pipeline.TAIL + 1.0)

        def sentinel(*a, **k):
            raise RuntimeError("recompile attempted")
        monkeypatch.setattr(pipeline, "_greedy_seed", sentinel)
        with pytest.raises(RuntimeError, match="recompile attempted"):
            pipeline.compile_pipeline(use_cache=True)

    def test_cache_key_covers_all_solve_constants(self):
        key = pipeline._cache_key(4)
        for field in ("bend", "dt", "c_speed", "tail", "lane_dt",
                      "w_time", "solve_rtol", "n_gates"):
            assert field in key


class TestCalibrationFailure:
    def test_calibrated_d_raises_without_sign_change(self, monkeypatch):
        monkeypatch.setattr(circuits, "_cal_d_cache", None)
        monkeypatch.setattr(circuits, "run_cascade",
                            lambda *a, **k: {"vel": np.ones((3, 1, 2)),
                                             "traj": np.ones((3, 1, 2)),
                                             "b_index": 1})
        with pytest.raises(RuntimeError, match="no sign change"):
            circuits.calibrated_d()
        monkeypatch.setattr(circuits, "_cal_d_cache", None)