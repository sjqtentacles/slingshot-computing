"""Render the demos as styled GIFs for the README (-> docs/*.gif).

Reads the trajectory JSON the demos already emit (out/*.json), so run the
demos first:  switch_demo, arithmetic_demo, pipeline_demo.

Usage: python -m demos.make_gifs   (from the repo root)
Needs Pillow; optionally uses ImageMagick `convert` to shrink the output.
"""

import json
import pathlib
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DOCS = ROOT / "docs"

GROUND = "#070a14"
GRID = "#141b36"
SIGNAL = "#54d1ff"
BIT = "#ffb454"
THIRD = "#b48cff"
GOOD = "#58e0a8"
OFF = "#2a3357"
DIM = "#6a7494"
TEXT = "#c9d2ea"
MONO = {"family": "monospace"}


def _new_ax(xlim, ylim, figsize=(6.4, 5.2)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=GROUND)
    ax.set_facecolor(GROUND)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for gx in range(int(np.floor(xlim[0] / 8)) * 8, int(xlim[1]) + 1, 8):
        ax.axvline(gx, color=GRID, lw=0.8, zorder=0)
    for gy in range(int(np.floor(ylim[0] / 8)) * 8, int(ylim[1]) + 1, 8):
        ax.axhline(gy, color=GRID, lw=0.8, zorder=0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, ax


def _starfield(ax, xlim, ylim, n=90, seed=7):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(*xlim, n)
    ys = rng.uniform(*ylim, n)
    ax.scatter(xs, ys, s=rng.uniform(0.3, 2.0, n), c="#9fb0dd",
               alpha=0.35, zorder=0, edgecolors="none")


def _fading_trail(ax, color, lw=2.4, tail=60):
    """A LineCollection whose segments fade from transparent to `color`."""
    lc = LineCollection([], linewidths=lw, capstyle="round", zorder=3)
    ax.add_collection(lc)
    head = ax.scatter([], [], s=90, c=color, zorder=4,
                      edgecolors="none")
    glow = ax.scatter([], [], s=360, c=color, alpha=0.28, zorder=3,
                      edgecolors="none")
    rgb = np.array(matplotlib.colors.to_rgb(color))

    def update(pts, upto):
        upto = min(upto, len(pts) - 1)
        lo = max(0, upto - tail)
        seg_pts = pts[lo:upto + 1]
        if len(seg_pts) >= 2:
            segs = np.stack([seg_pts[:-1], seg_pts[1:]], axis=1)
            a = np.linspace(0.05, 1.0, len(segs))
            lc.set_segments(segs)
            lc.set_color(np.column_stack([np.tile(rgb, (len(segs), 1)), a]))
        else:
            lc.set_segments([])
        p = pts[upto]
        head.set_offsets([p])
        glow.set_offsets([p])
    return update


def _save(fig, anim, name, fps, dpi=100):
    DOCS.mkdir(exist_ok=True)
    path = DOCS / name
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi,
              savefig_kwargs={"facecolor": GROUND})
    plt.close(fig)
    _shrink(path)
    print(f"wrote {path}  ({path.stat().st_size // 1024} KB)")


def _shrink(path):
    if not shutil.which("convert"):
        return
    tmp = path.with_suffix(".opt.gif")
    try:
        subprocess.run(
            ["convert", str(path), "-layers", "optimize", "-fuzz", "3%",
             str(tmp)], check=True, capture_output=True)
        if tmp.stat().st_size < path.stat().st_size:
            tmp.replace(path)
        else:
            tmp.unlink()
    except subprocess.CalledProcessError:
        if tmp.exists():
            tmp.unlink()


# --------------------------------------------------------------------------
# 1. flagship pipeline — cycle representative inputs to show the encoder
# --------------------------------------------------------------------------

def gif_pipeline(play=34, hold=9):
    data = json.loads((OUT / "pipeline.json").read_text())
    N = data["spec"]["n_gates"]
    show = ["1111", "1100", "1000", "0000"]  # AND, P2, P1, straight-through
    show = [s[:N] for s in show]

    # port anchors + world window from B lanes of shown cases
    def bpath(key):
        return np.array(data["cases"][key]["bodies"][0])
    port_anchor = {}
    for k in range(N + 1):
        key = "".join("1" if i < k else "0" for i in range(N))
        port_anchor[k] = bpath(key)[-1]
    allpts = np.concatenate([bpath(k) for k in show] + [np.array(list(port_anchor.values()))])
    pad = 3
    xlim = (allpts[:, 0].min() - pad, allpts[:, 0].max() + pad)
    ylim = (allpts[:, 1].min() - pad, allpts[:, 1].max() + pad)

    fig, ax = _new_ax(xlim, ylim, figsize=(6.6, 5.4))
    _starfield(ax, xlim, ylim)

    # static ghost fan of every port lane
    for k in range(N + 1):
        key = "".join("1" if i < k else "0" for i in range(N))
        p = bpath(key)
        ax.plot(p[:, 0], p[:, 1], color=SIGNAL, lw=1.0, alpha=0.10,
                ls=(0, (2, 4)), zorder=1)

    port_rings, port_labels = [], []
    for k in range(N + 1):
        x, y = port_anchor[k]
        ring = plt.Circle((x, y), 1.7, fill=False, ec=OFF, lw=1.5,
                          ls=(0, (3, 3)), zorder=2)
        ax.add_patch(ring)
        port_rings.append(ring)
        port_labels.append(ax.text(x, y + 2.6, f"P{k}", color=DIM,
                                   ha="center", fontsize=11, **MONO))

    b_trail = _fading_trail(ax, SIGNAL, lw=3.0, tail=70)
    ctrl_trails = [_fading_trail(ax, BIT, lw=2.4, tail=55) for _ in range(N)]
    in_txt = ax.text(0.03, 0.95, "", transform=ax.transAxes, color=BIT,
                     fontsize=17, va="top", weight="bold", **MONO)
    out_txt = ax.text(0.03, 0.85, "", transform=ax.transAxes, color=TEXT,
                      fontsize=15, va="top", weight="bold", **MONO)
    ax.text(0.03, 0.045, "gravitational pipeline · 4-input priority encoder",
            transform=ax.transAxes, color=DIM, fontsize=9, **MONO)

    # build a frame plan: (case_key, frame_index_in_case, done)
    plan = []
    for key in show:
        nf = len(bpath(key))
        idxs = np.linspace(0, nf - 1, play).astype(int)
        for j, fi in enumerate(idxs):
            plan.append((key, fi, j == play - 1))
        plan += [(key, nf - 1, True)] * hold

    def draw(frame):
        key, fi, done = plan[frame]
        case = data["cases"][key]
        present = case["present"]
        b_trail(bpath(key), fi)
        for ci in range(N):
            if ci < len(present):
                ctrl_trails[ci](np.array(case["bodies"][1 + ci]), fi)
            else:
                ctrl_trails[ci](np.array([[1e6, 1e6], [1e6, 1e6]]), 1)
        port = case["port"]
        for k, ring in enumerate(port_rings):
            lit = done and k == port
            col = (GOOD if k == N else BIT) if lit else OFF
            ring.set_edgecolor(col)
            ring.set_linewidth(3.2 if lit else 1.5)
            ring.set_linestyle("solid" if lit else (0, (3, 3)))
            port_labels[k].set_color(col if lit else DIM)
        bits = " ".join(key)
        in_txt.set_text(f"in  {bits}")
        if done:
            tag = "  AND" if port == N else ""
            out_txt.set_text(f"out P{port}{tag}")
            out_txt.set_color(GOOD if port == N else BIT)
        else:
            out_txt.set_text("out ··")
            out_txt.set_color(DIM)
        return []

    anim = FuncAnimation(fig, draw, frames=len(plan), interval=55, blit=False)
    _save(fig, anim, "pipeline.gif", fps=20)


# --------------------------------------------------------------------------
# 2. the SWITCH primitive — A absent vs present
# --------------------------------------------------------------------------

def gif_switch(play=42, hold=10):
    data = json.loads((OUT / "switch.json").read_text())
    runs = [("absent", data["absent"]), ("present", data["present"])]
    allpts = []
    for _, r in runs:
        for b in r["bodies"]:
            allpts += b
    allpts = np.array(allpts)
    pad = 3
    xlim = (allpts[:, 0].min() - pad, allpts[:, 0].max() + pad)
    ylim = (allpts[:, 1].min() - pad, allpts[:, 1].max() + pad + 4)  # label headroom

    fig, ax = _new_ax(xlim, ylim, figsize=(6.6, 4.9))
    _starfield(ax, xlim, ylim, seed=3)

    for _, r in runs:  # faint ghosts of both signal paths
        bp = np.array(r["bodies"][r["b_index"]])
        ax.plot(bp[:, 0], bp[:, 1], color=SIGNAL, lw=1.0, alpha=0.10,
                ls=(0, (2, 4)), zorder=1)

    p0 = np.array(data["absent"]["bodies"][data["absent"]["b_index"]])[-1]
    p1 = np.array(data["present"]["bodies"][data["present"]["b_index"]])[-1]
    rings = []
    for (x, y), lab in [(p0, "PORT 0"), (p1, "PORT 1")]:
        ring = plt.Circle((x, y), 1.4, fill=False, ec=OFF, lw=1.5,
                          ls=(0, (3, 3)), zorder=2)
        ax.add_patch(ring)
        rings.append(ring)
        ax.text(x, y + 2.2, lab, color=DIM, ha="center", fontsize=10, **MONO)

    b_trail = _fading_trail(ax, SIGNAL, lw=3.0, tail=70)
    a_trail = _fading_trail(ax, BIT, lw=2.6, tail=70)
    label = ax.text(0.03, 0.94, "", transform=ax.transAxes, fontsize=15,
                    va="top", weight="bold", **MONO)
    ax.text(0.03, 0.05, "the SWITCH gate · one bit from one flyby",
            transform=ax.transAxes, color=DIM, fontsize=9, **MONO)

    plan = []
    for key, r in runs:
        nf = len(r["bodies"][0])
        for fi in np.linspace(0, nf - 1, play).astype(int):
            plan.append((key, int(fi), False))
        plan += [(key, nf - 1, True)] * hold

    def draw(frame):
        key, fi, done = plan[frame]
        r = data[key]
        bi = r["b_index"]
        b_trail(np.array(r["bodies"][bi]), fi)
        if key == "present":
            a_trail(np.array(r["bodies"][0]), fi)
        else:
            a_trail(np.array([[1e6, 1e6], [1e6, 1e6]]), 1)
        port = r["port"]
        for k, ring in enumerate(rings):
            lit = done and k == port
            ring.set_edgecolor(SIGNAL if lit else OFF)
            ring.set_linewidth(3.2 if lit else 1.5)
            ring.set_linestyle("solid" if lit else (0, (3, 3)))
        label.set_color(BIT if key == "present" else DIM)
        label.set_text(f"A = {1 if key == 'present' else 0}"
                       + (f"   ->  PORT {port}" if done else ""))
        return []

    anim = FuncAnimation(fig, draw, frames=len(plan), interval=45, blit=False)
    _save(fig, anim, "switch.gif", fps=22)


# --------------------------------------------------------------------------
# 3. the half adder — 1 + 1 = 10
# --------------------------------------------------------------------------

def gif_adder(play=40, hold=12):
    data = json.loads((OUT / "arithmetic.json").read_text())["adder"]
    order = ["01", "10", "11"]
    allpts = []
    for key in order:
        for b in data[key]["bodies"]:
            allpts += b
    allpts = np.array(allpts)
    pad = 2.5
    xlim = (allpts[:, 0].min() - pad, allpts[:, 0].max() + pad)
    ylim = (allpts[:, 1].min() - pad, allpts[:, 1].max() + pad + 3.5)  # label headroom

    fig, ax = _new_ax(xlim, ylim, figsize=(6.6, 5.2))
    _starfield(ax, xlim, ylim, seed=5)

    sum_ports = [(12.0, 0.5), (-12.0, -0.5)]
    carry_ports = [(-8.3, 10.1), (8.3, -10.1)]
    for (x, y) in sum_ports:
        ax.add_patch(plt.Circle((x, y), 1.6, fill=False, ec=OFF, lw=1.2,
                                ls=(0, (3, 3)), zorder=2))
        ax.text(x, y + 2.3, "SUM", color=DIM, ha="center", fontsize=9, **MONO)
    for (x, y) in carry_ports:
        ax.add_patch(plt.Circle((x, y), 1.8, fill=False, ec=OFF, lw=1.2,
                                ls=(0, (3, 3)), zorder=2))
        ax.text(x, y + 2.5, "CARRY", color=DIM, ha="center", fontsize=9, **MONO)

    a_trail = _fading_trail(ax, BIT, lw=2.8, tail=70)
    b_trail = _fading_trail(ax, SIGNAL, lw=2.8, tail=70)
    label = ax.text(0.03, 0.94, "", transform=ax.transAxes, fontsize=17,
                    va="top", weight="bold", color=TEXT, **MONO)
    ax.text(0.03, 0.05, "the half adder · gravity adds two bits",
            transform=ax.transAxes, color=DIM, fontsize=9, **MONO)

    plan = []
    for key in order:
        nf = len(data[key]["bodies"][0])
        for fi in np.linspace(0, nf - 1, play).astype(int):
            plan.append((key, int(fi), False))
        plan += [(key, nf - 1, True)] * hold

    def draw(frame):
        key, fi, done = plan[frame]
        case = data[key]
        names = case["names"]
        far = np.array([[1e6, 1e6], [1e6, 1e6]])
        a_trail(np.array(case["bodies"][names.index("A")]) if "A" in names else far,
                fi if "A" in names else 1)
        b_trail(np.array(case["bodies"][names.index("B")]) if "B" in names else far,
                fi if "B" in names else 1)
        eq = f"{key[0]} + {key[1]}"
        if done:
            eq += f" = {case['carry']}{case['sum']}"
        label.set_text(eq)
        label.set_color(GOOD if (done and key == "11") else TEXT)
        return []

    anim = FuncAnimation(fig, draw, frames=len(plan), interval=48, blit=False)
    _save(fig, anim, "adder.gif", fps=21)


def main():
    gif_pipeline()
    gif_switch()
    gif_adder()


if __name__ == "__main__":
    main()
