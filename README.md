# Slingshot Computing

Logic gates built from nothing but Newtonian point-mass gravity. Bits are the
presence/absence of small bodies on ballistic trajectories; gates are close
hyperbolic flybys that deflect them between output ports — the gravitational
analogue of the Fredkin–Toffoli billiard-ball computer, with elastic collisions
replaced by two-body Kepler scattering.

## The primitive: the SWITCH gate

Ball B (the signal) always flies. Bit A is whether ball A is launched.

- A absent → B flies straight → **port 0**
- A present → mutual gravitational deflection (impact parameter b=1, relative
  speed v=2, so tan(θ/2) = G(m_A+m_B)/(b·v²) = 0.5 → θ ≈ 53°) → **port 1**

A switch is universal-adjacent: switches + routing (slingshot "mirrors" around
heavy masses) give AND/OR/NOT the same way the billiard-ball model does.

## Demo 02: gravity does arithmetic

`slingshot/circuits.py` builds two composed circuits on the primitive:

- **Half adder** — the flyby is Fredkin–Toffoli's *interaction gate*: with both
  balls as input bits, deflected exits = CARRY (A∧B), straight exits = SUM
  (A⊕B at the detector level). All four cases verified: 1+1=10.
- **Gate cascade** — ball B always flies; bit A gates flyby 1, bit C is a third
  ball aimed at B's *post-gate-1* trajectory. B ends double-bent iff A∧C.
  Composition is the step that makes ballistic logic a computer.

Two lessons the cascade forced (both now measured, see `demos/arithmetic_demo.py`):

- **No shielding**: C's long-range pull drags B off its gate-1 course during the
  whole approach — the naive billiard aim collapses the impact parameter from
  1.0 to 0.1. Wires must be calibrated on the full three-body problem
  (`circuits.calibrated_d()`, root-found on the simulation).
- **No insulation**: deflection falls off only as ~1/b, so idle lanes bend each
  other (7° crosstalk on the straight lane). Gate 2 had to move downstream
  until the logic lanes diverged ~9 units before C's wire could cross safely.

**The chaos tax, measured**: a 1e-8 launch offset grows ×6 through gate 1 and
×17 through gate 2 — ~1 digit of precision per gate, no restoring force. Float64
affords ~12 gates at this scale (`out/chaos.png`).

## Demo 03 (flagship): the 4-gate pipeline

`slingshot/pipeline.py` compiles a **5-body machine**: one signal ball B threads
**four gravitational gates in series**. Each present control bends B one 54°
port-step deeper, so B's exit port = **index of the first absent control** — a
**4-input priority encoder**, and all-present is a **4-input AND** at the deepest
port. All 16 input cases verified (worst decision margin 15.4° vs a 27° boundary).

The compiler is the interesting part: the five bodies form one coupled
system (no gravitational shielding), so gates can't be calibrated independently —
naive per-gate loops limit-cycle. It's solved as a **boundary-value problem**:
a greedy seed, then Levenberg–Marquardt over per-gate (timing, impact-parameter)
knobs with a **local per-gate bend residual** (a cumulative target lets adjacent
gates split a port — the 27° = BEND/2 trap; the local target forbids it).

**The depth wall (a result, not a bug):** four gates is the ceiling. A 5th
control must fly ~60 units through the whole accumulated field, is chaotically
deflected, and can't be aimed onto B — its calibration Jacobian goes flat.
Gates 1–4 compile every time; gate 5 never lands. **Chaos bounds computational
depth, not just precision** — and the two limits are separable: the measured
chaos tax (~1.1 digits/gate, 4.4 over the chain) would allow ~13 gates on
float64, so *aiming*, not precision, is what caps this design at ~5.

## Run it

```
python3 -m demos.switch_demo       # demo 01: SWITCH gate
python3 -m demos.arithmetic_demo   # demo 02: half adder + cascade + chaos tax
python3 -m demos.pipeline_demo     # demo 03: 4-gate priority encoder + AND (flagship)
python3 -m demos.build_viewer      # build the animated out/*.html viewers
python3 -m pytest                  # 56-test suite (physics invariants + all truth tables)
```

Each demo prints its truth table / measurements and writes plots + trajectory
JSON to `out/` for the animated HTML viewers. The pipeline spec is cached in
`out/pipeline_spec.json` (compilation takes a few minutes; delete to recompile).

## Physics / numerics

- Pure pairwise Newtonian gravity, G = 1, planar, **no softening** (softening
  would smear out the sharp scattering the gate depends on).
- Integrated with adaptive high-order Runge–Kutta (scipy DOP853,
  rtol = atol = 1e-12). Energy drift on the demo: ~5e-13.

## Honest caveats / prior art

- Turing completeness here holds only in the exact-real idealization: n-body
  chaos consumes ~constant digits of initial-condition precision per gate, and
  gravity has no attractor states, so there's no error correction. That decay
  is a feature of the project — measure it, don't hide it.
- Nearest prior art: Fredkin & Toffoli's billiard-ball computer (1982), Moore's
  Turing-machine embeddings in smooth dynamics (1990), Cardona–Miranda–
  Peralta-Salas–Presas on Turing-complete Euler flows (2021), Tao's
  universality-of-dynamics program. Gravitational-slingshot gates as a built
  artifact appear to be unclaimed territory.

## Roadmap

- [x] SWITCH gate (one flyby, two ports)
- [x] AND gate (cascade: two flybys composed in series)
- [x] Half adder (interaction gate + detector regions)
- [x] Measure bits-of-precision consumed per gate
- [x] Circuit compiler: N gates auto-placed and calibrated (boundary-value solve)
- [x] 4-gate pipeline: priority encoder + 4-input AND, 16 cases
- [x] Found the depth wall (~4 gates) set by chaotic control-flight
- [ ] Heavy "mirror" controls (near-ballistic) to push depth past 4
- [ ] Dual-rail encoding so absence-of-ball isn't the only "0"
- [ ] Fan-out / signal copying (the hard one: no cloning in reversible ballistics)
