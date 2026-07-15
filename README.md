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

## Run it

```
python3 -m demos.switch_demo
```

Prints the port classification and energy drift for both runs, writes
`out/switch.png` (trajectory plot) and `out/switch.json` (trajectories for the
animated HTML viewer).

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
- [ ] Slingshot mirror: route a signal around a heavy fixed mass
- [ ] Compose two gates; measure bits-of-precision consumed per gate vs. λt
- [ ] AND gate from switch + routing
- [ ] Dual-rail encoding so absence-of-ball isn't the only "0"
