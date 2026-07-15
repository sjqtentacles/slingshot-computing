"""From-scratch compilation, isolated from the shared cache.

The main suite deliberately reuses the cached spec (the compile is minutes);
this slow-marked test proves the compiler itself still converges from zero —
on a smaller 2-gate machine so it stays ~a minute.
"""

import pytest

from slingshot import pipeline


@pytest.mark.slow
def test_two_gate_machine_compiles_from_scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SPEC_CACHE", tmp_path / "spec.json")
    spec = pipeline.compile_pipeline(n_gates=2, use_cache=True)
    assert spec["n_gates"] == 2
    assert len(spec["ports"]) == 3
    # ports track the design fan and stay separated
    for k, p in enumerate(spec["ports"]):
        assert abs(pipeline.wrap_deg(p - pipeline.ideal_port_heading(k))) < 10.0
    # and the freshly compiled machine actually computes
    res = pipeline.run_pipeline(spec, (1, 1))
    k, margin = pipeline.classify_pipeline(spec, res)
    assert k == 2 and margin < 18.0
    res0 = pipeline.run_pipeline(spec, (0, 1))
    assert pipeline.classify_pipeline(spec, res0)[0] == 0
