"""Gravity does arithmetic: half adder + two-gate cascade + chaos tax.

Runs every input case of both circuits, asserts the truth tables, measures
how fast a tiny input error grows through the gate cascade, and writes
out/arithmetic.json (for the HTML viewer) + out/chaos.png.

Usage: python -m demos.arithmetic_demo  (from the repo root)
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slingshot import circuits

OUT = pathlib.Path(__file__).resolve().parent.parent / "out"

CASCADE_OUT = {"straight": 0, "bend1": 0, "bend2": 1}


def case_payload(res, extra):
    return {
        "t": np.round(res["t"], 4).tolist(),
        "names": res["names"],
        "bodies": [np.round(res["traj"][i], 4).tolist()
                   for i in range(len(res["names"]))],
        "energy_drift": float(res["energy_drift"]),
        **extra,
    }


def main():
    OUT.mkdir(exist_ok=True)
    payload = {"adder": {}, "cascade": {}}

    print("=== half adder:  A + B  ->  CARRY SUM ===")
    for a in (0, 1):
        for b in (0, 1):
            res = circuits.run_half_adder(a, b)
            s, cy = circuits.classify_half_adder(res)
            assert (s, cy) == ((a + b) % 2, (a + b) // 2), f"adder failed at {a}+{b}"
            print(f"  {a} + {b} = {cy}{s}")
            payload["adder"][f"{a}{b}"] = case_payload(res, {"sum": s, "carry": cy})
    print("  truth table correct: gravity adds bits (1 + 1 = 10).")

    print("\n=== cascade:  B double-bends iff A AND C ===")
    for a in (0, 1):
        for c in (0, 1):
            res = circuits.run_cascade(a, c)
            port = circuits.classify_cascade(res)
            out = CASCADE_OUT[port]
            assert out == (a & c), f"cascade failed at A={a} C={c} (port {port})"
            print(f"  A={a} C={c}  ->  {port:8s}  out={out}  "
                  f"(drift {res['energy_drift']:.1e})")
            payload["cascade"][f"{a}{c}"] = case_payload(
                res, {"out": out, "port": port,
                      "b_index": res["b_index"]})
    print("  truth table correct: two flybys compose into AND.")

    print("\n=== chaos tax: error growth through the cascade (A=1, C=1) ===")
    delta = 1e-8
    nom = circuits.run_cascade(1, 1, n_samples=1400)
    pert = circuits.run_cascade(1, 1, n_samples=1400, perturb_b=delta)
    bi = nom["b_index"]
    sep = np.linalg.norm(nom["traj"][bi] - pert["traj"][bi], axis=1)
    t = nom["t"]

    def sep_at(tq):
        return float(np.interp(tq, t, sep))

    s_in, s_g1, s_g2 = delta, sep_at(16.0), sep_at(circuits.T_END)
    amp1, amp2 = s_g1 / s_in, s_g2 / s_g1
    print(f"  input offset          {delta:.1e}")
    print(f"  after gate 1 (t=16)   {s_g1:.3e}   x{amp1:,.0f} = "
          f"{np.log10(amp1):.1f} digits eaten")
    print(f"  after gate 2 (t=31)   {s_g2:.3e}   x{amp2:,.0f} = "
          f"{np.log10(amp2):.1f} digits eaten")
    print(f"  float64 carries ~15-16 digits -> "
          f"~{int(15 / max(np.log10(amp1), np.log10(amp2)))} gates max at this scale")

    payload["chaos"] = {
        "delta": delta,
        "t": np.round(t[::4], 3).tolist(),
        "sep": [float(f"{v:.4e}") for v in sep[::4]],
        "after_gate1": s_g1, "after_gate2": s_g2,
        "digits_gate1": float(np.log10(amp1)), "digits_gate2": float(np.log10(amp2)),
    }

    (OUT / "arithmetic.json").write_text(json.dumps(payload))
    print(f"\nwrote {OUT / 'arithmetic.json'}")

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0b0e1a")
    ax.set_facecolor("#0b0e1a")
    ax.semilogy(t, np.maximum(sep, 1e-13), color="#54d1ff", lw=2)
    for tg, label in ((10.0, "gate 1 flyby"), (circuits.T2, "gate 2 flyby")):
        ax.axvline(tg, color="#ffb454", ls="--", lw=1, alpha=0.7)
        ax.text(tg, 0.03, label, color="#ffb454", fontsize=9, rotation=90,
                transform=ax.get_xaxis_transform(), ha="right", va="bottom")
    ax.set_xlabel("time", color="w")
    ax.set_ylabel("|Δ position of ball B|  (input offset 1e-8)", color="w")
    ax.set_title("each flyby multiplies the input error — the chaos tax per gate",
                 color="w")
    ax.tick_params(colors="#889")
    for s in ax.spines.values():
        s.set_color("#334")
    fig.tight_layout()
    fig.savefig(OUT / "chaos.png", dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {OUT / 'chaos.png'}")


if __name__ == "__main__":
    main()
