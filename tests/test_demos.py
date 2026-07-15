"""Smoke coverage for the demo layer: imports, entry points, and the
viewer-payload contract (the JSON keys the committed HTML viewers consume).
Full demo runs render figures for minutes; that stays out of the suite."""

import importlib
import json

import pytest

DEMOS = ["switch_demo", "arithmetic_demo", "pipeline_demo", "build_viewer",
         "make_gifs"]


@pytest.mark.parametrize("name", DEMOS)
def test_demo_imports(name):
    assert importlib.import_module(f"demos.{name}") is not None


def test_demo_entry_points_exist():
    from demos import (arithmetic_demo, build_viewer, make_gifs,
                       pipeline_demo, switch_demo)
    for mod in (switch_demo, arithmetic_demo, pipeline_demo, build_viewer,
                make_gifs):
        assert callable(mod.main)


def test_build_viewer_injects_payload(tmp_path, monkeypatch):
    """The injection mechanic the HTML viewers depend on: the JSON payload
    replaces the placeholder, exactly once."""
    from demos import build_viewer
    template = tmp_path / "t.html"
    template.write_text("<script>const DATA = /*__DATA__*/;</script>")
    data = tmp_path / "d.json"
    payload = {"cases": {"11": {"bodies": [[[0, 0]]], "present": [0]}},
               "spec": {"ports": [180.0], "n_gates": 1}}
    data.write_text(json.dumps(payload))
    out = tmp_path / "out.html"
    monkeypatch.setattr(build_viewer, "ROOT", tmp_path)
    monkeypatch.setattr(build_viewer, "PAGES", [("t.html", "d.json", "out.html")])
    build_viewer.main()
    html = out.read_text()
    assert "/*__DATA__*/" not in html
    assert '"ports": [180.0]' in html


def test_pipeline_payload_schema():
    """pipeline_demo's JSON must keep the keys pipeline.html consumes.
    Uses the compiled spec (cached) and one cheap run instead of the demo's
    full 16-case sweep."""
    import numpy as np

    from slingshot import pipeline
    spec = pipeline.compile_pipeline()
    res = pipeline.run_pipeline(spec, (1,) * spec["n_gates"], n_samples=200)
    k, margin = pipeline.classify_pipeline(spec, res)
    case = {"bits": [1] * spec["n_gates"], "port": k,
            "margin": round(margin, 2), "is_and": True,
            "bodies": [np.round(res["traj"][i], 3).tolist()
                       for i in range(res["traj"].shape[0])],
            "present": list(range(spec["n_gates"]))}
    # the contract pipeline.html relies on
    assert {"bits", "port", "bodies", "present"} <= set(case)
    assert {"ports", "n_gates", "gates", "t_end"} <= set(spec)