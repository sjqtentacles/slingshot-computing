"""The flagship: a 5-body, 4-gate gravitational pipeline.

One signal ball threads four calibrated gravitational gates in series. The
same machine computes, simultaneously:
  - a 4-input priority encoder: B's exit port = index of the first absent
    control (the first 0 bit),
  - a 4-input AND: all controls present -> the deepest port,
and answers empirically "how deep can a float64 gravity computer run" by
measuring the precision eaten per gate.

Four gates is the depth ceiling of this launch-from-afar design: a 5th
control can't be aimed through the accumulated field (see slingshot.pipeline).

Runs all 16 input subsets, verifies the truth table, measures the chaos tax
vs depth, and writes out/pipeline.json + the chaos figure to BOTH
out/pipeline_chaos.png and docs/pipeline_chaos.png (the README embeds the
latter, so it must be reproducible from this script).

Usage: python -m demos.pipeline_demo   (from the repo root)
"""

import itertools
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slingshot import pipeline

OUT = pathlib.Path(__file__).resolve().parent.parent / "out"
N = 4


def first_zero(bits):
    return bits.index(0) if 0 in bits else len(bits)


def main():
    OUT.mkdir(exist_ok=True)
    print(f"compiling {N}-gate machine (global LM solve)...")
    spec = pipeline.compile_pipeline(n_gates=N)
    print(f"  ports (deg): {[round(p, 1) for p in spec['ports']]}")
    print(f"  impact params: {[round(g['b'], 2) for g in spec['gates']]}")
    print(f"  solve: worst {spec['solve']['worst_deg']:.2f} deg, "
          f"{spec['solve']['nfev']} sims\n")

    n_cases = 2 ** N
    print(f"=== priority encoder / {N}-input AND: all {n_cases} inputs ===")
    payload_cases = {}
    worst_margin = 0.0
    for bits in itertools.product((0, 1), repeat=N):
        res = pipeline.run_pipeline(spec, bits, n_samples=700)
        k, margin = pipeline.classify_pipeline(spec, res)
        expect = first_zero(list(bits))
        ok = k == expect
        worst_margin = max(worst_margin, margin)
        flag = "" if ok else "  <-- WRONG"
        assert ok, f"bits={bits}: routed to port {k}, expected {expect}{flag}"
        key = "".join(map(str, bits))
        payload_cases[key] = {
            "bits": list(bits), "port": k, "margin": round(margin, 2),
            "is_and": all(bits),
            "bodies": [np.round(res["traj"][i], 3).tolist()
                       for i in range(res["traj"].shape[0])],
            "present": [i for i, b in enumerate(bits) if b],
        }
    print(f"  all {n_cases} cases correct. worst decision margin "
          f"{worst_margin:.1f} deg (boundary is {pipeline.BEND / 2:.0f}).")
    print(f"  1111 -> port {N} (AND); 0... -> port 0; 1110 -> port 3; etc.\n")

    print("=== chaos tax vs depth (all-ones) ===")
    delta = 1e-9
    nom = pipeline.run_pipeline(spec, (1,) * N, n_samples=2500)
    pert = pipeline.run_pipeline(spec, (1,) * N, n_samples=2500, perturb_b=delta)
    sep = np.linalg.norm(nom["traj"][0] - pert["traj"][0], axis=1)
    t = nom["t"]
    per_gate = []
    prev = delta
    for k in range(N):
        s = float(np.interp(spec["gates"][k]["t"] + 5.0, t, sep))
        digits = np.log10(s / prev)
        per_gate.append({"gate": k + 1, "sep": s, "digits": float(digits)})
        print(f"  after gate {k + 1}: sep {s:.2e}  (+{digits:.2f} digits)")
        prev = s
    total_digits = np.log10(sep[-1] / delta)
    max_gates = int(15 / (total_digits / N))
    print(f"  total {total_digits:.1f} digits over {N} gates -> "
          f"~{max_gates} gates on float64's ~15-digit budget.\n")

    payload = {
        "spec": {"ports": spec["ports"], "t_end": spec["t_end"],
                 "gates": spec["gates"], "bend": spec["bend"],
                 "n_gates": N, "solve": spec["solve"]},
        "cases": payload_cases,
        "chaos": {"delta": delta, "t": np.round(t[::5], 3).tolist(),
                  "sep": [float(f"{v:.4e}") for v in sep[::5]],
                  "per_gate": per_gate, "total_digits": float(total_digits),
                  "max_gates": max_gates},
        "worst_margin": round(worst_margin, 2),
    }
    (OUT / "pipeline.json").write_text(json.dumps(payload))
    print(f"wrote {OUT / 'pipeline.json'}")

    # chaos-vs-depth figure
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#070a14")
    ax.set_facecolor("#070a14")
    ax.semilogy(t, np.maximum(sep, 1e-13), color="#54d1ff", lw=2)
    for k in range(N):
        tg = spec["gates"][k]["t"]
        ax.axvline(tg, color="#ffb454", ls="--", lw=1, alpha=0.6)
        ax.text(tg, 0.02, f"gate {k + 1}", color="#ffb454", fontsize=8,
                rotation=90, transform=ax.get_xaxis_transform(),
                ha="right", va="bottom")
    ax.set_xlabel("time", color="w")
    ax.set_ylabel(f"|Δ position of B|   (launch offset {delta:.0e})", color="w")
    ax.set_title(f"chaos tax vs depth — {N} gates, "
                 f"{total_digits:.1f} digits of precision consumed", color="w")
    ax.tick_params(colors="#889")
    for s in ax.spines.values():
        s.set_color("#334")
    fig.tight_layout()
    docs = OUT.parent / "docs"
    docs.mkdir(exist_ok=True)
    for target in (OUT / "pipeline_chaos.png", docs / "pipeline_chaos.png"):
        fig.savefig(target, dpi=150, facecolor=fig.get_facecolor())
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
