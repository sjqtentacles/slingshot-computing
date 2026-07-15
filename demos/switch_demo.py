"""Run the gravitational SWITCH gate both ways, plot it, dump JSON for viewers.

Usage: python -m demos.switch_demo  (from the repo root)
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slingshot import gates

OUT = pathlib.Path(__file__).resolve().parent.parent / "out"


def main():
    OUT.mkdir(exist_ok=True)
    runs = {}
    for a_present in (False, True):
        res = gates.run_switch(a_present)
        port, angle = gates.classify_port(res)
        runs[a_present] = (res, port, angle)
        label = "A present" if a_present else "A absent "
        print(
            f"{label}: B exits port {port}  (exit angle {angle:+6.2f} deg, "
            f"energy drift {res['energy_drift']:.2e})"
        )

    port0, port1 = runs[False][1], runs[True][1]
    assert port0 == 0 and port1 == 1, "gate failed to switch!"
    print("\nSWITCH gate works: bit A routes ball B between ports 0 and 1.")

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#0b0e1a")
    titles = ["A absent  ->  B exits port 0", "A present  ->  B exits port 1"]
    for ax, a_present, title in zip(axes, (False, True), titles):
        res, port, _ = runs[a_present]
        ax.set_facecolor("#0b0e1a")
        ax.set_title(title, color="w", fontsize=12)
        if a_present:
            ax.plot(*res["traj"][0].T, color="#ff9f43", lw=1.8, label="ball A (bit)")
            ax.plot(*res["traj"][0, -1], "o", color="#ff9f43", ms=8)
        bi = res["b_index"]
        ax.plot(*res["traj"][bi].T, color="#54d1ff", lw=1.8, label="ball B (signal)")
        ax.plot(*res["traj"][bi, -1], "o", color="#54d1ff", ms=8)
        ax.annotate(
            f"port {port}", res["traj"][bi, -1], textcoords="offset points",
            xytext=(10, 10), color="w", fontsize=11,
        )
        ax.set_xlim(-16, 16)
        ax.set_ylim(-9, 13)
        ax.set_aspect("equal")
        ax.tick_params(colors="#556")
        for s in ax.spines.values():
            s.set_color("#334")
        ax.legend(facecolor="#141830", labelcolor="w", edgecolor="#334")
    fig.suptitle("Gravitational SWITCH gate — pure Newtonian point-mass gravity, G=1",
                 color="w", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "switch.png", dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {OUT / 'switch.png'}")

    # --- JSON for the HTML viewer ---
    payload = {}
    for a_present, key in ((False, "absent"), (True, "present")):
        res, port, angle = runs[a_present]
        payload[key] = {
            "t": res["t"].tolist(),
            "bodies": [np.round(res["traj"][i], 5).tolist() for i in range(len(res["masses"]))],
            "b_index": res["b_index"],
            "port": port,
            "exit_angle_deg": round(float(angle), 3),
            "energy_drift": float(res["energy_drift"]),
        }
    (OUT / "switch.json").write_text(json.dumps(payload))
    print(f"wrote {OUT / 'switch.json'}")


if __name__ == "__main__":
    main()
