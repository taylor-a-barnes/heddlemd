# OpenMM analogue of `spc-water-8192`

An OpenMM port of [`../spc-water-8192`](../spc-water-8192/), set up to
make wall-clock comparisons between the two engines on the *same
system*: 8192 rigid SPC/E water molecules (24,576 atoms), SPME
electrostatics, NVT equilibration → NPT production.

Reads `../spc-water-8192/water.in.xyz` directly, so both engines start
from identical positions and an identical box.

## Run

```
python3 run.py
```

Prints per-phase wall-clock time and ms/step for the three phases
(`min`, `equil`, `prod`) on stdout. Compare directly against the
`.timings` files written by `heddlemd run water.in.toml` in
`../spc-water-8192/`.

Useful flags:

| Flag             | Default | Effect                                       |
|------------------|---------|----------------------------------------------|
| `--platform`     | `CUDA`  | `CUDA`, `CPU`, or `Reference`.               |
| `--precision`    | `mixed` | `single`, `mixed`, `double` (CUDA only).     |
| `--equil-steps`  | 2500    | NVT steps (2 fs each).                       |
| `--prod-steps`   | 1000    | NPT steps (2 fs each).                       |
| `--seed`         | 1       | Maxwell-Boltzmann velocity sampling seed.    |

`--precision single` is the closest analogue to HeddleMD's f32-end-to-end
storage; `mixed` is OpenMM's standard production setting (forces in f32,
accumulators in f64).

## What is kept identical

- Initial positions and box, read from `../spc-water-8192/water.in.xyz`.
- Masses, charges, LJ sigma/epsilon, and rigid SPC/E geometry — copied
  verbatim from `../spc-water-8192/water.in.toml` (kg, C, m, J).
- SPME real-space cutoff (1 nm), Ewald parameter α (3.5 nm⁻¹), and FFT
  grid `(48, 48, 96)`.
- Integration timestep (2 fs), temperature (298.15 K), pressure
  (1.013×10⁵ Pa), and the 5 ps NVT + 2 ps NPT schedule.
- All three intramolecular pair constraints (O–H, O–H, H–H).

## What necessarily differs

The microscopic trajectories will not match — they are different
stochastic dynamics — but each step does comparable work:

| Setting     | HeddleMD                | OpenMM analogue                |
|-------------|-------------------------|--------------------------------|
| Thermostat  | CSVR (Bussi-Donadio)    | `LangevinMiddleIntegrator`     |
| Barostat    | c-rescale (Bernetti)    | `MonteCarloBarostat`           |
| Constraints | SHAKE (3 pair)          | SETTLE (auto-detected)         |
| Storage     | f32 end-to-end          | Mixed (f32 forces, f64 sums)   |

OpenMM does not ship CSVR or c-rescale. The Langevin / MC pair is the
standard OpenMM idiom for NVT / NPT, samples the same target
ensembles, and has comparable per-step cost. OpenMM detects the
water-shape constraint triple automatically and dispatches to the
analytic SETTLE solver, which is faster than HeddleMD's iterative
SHAKE — this is a legitimate engine-level difference and part of what
the benchmark is measuring.

## Expected output

On a modern NVIDIA GPU you should see something like:

```
loaded 24576 atoms (8192 waters) from .../water.in.xyz
box: 5.019 x 5.019 x 10.038 nm
platform: CUDA (precision=mixed)
[min]   wall =   ~1 s     U = -4.27e+05 kJ/mol
[equil] wall =   ~1 s     steps =  2500    ~0.4 ms/step
[prod]  wall =   ~0.5 s   steps =  1000    ~0.5 ms/step
```

The minimized potential energy of ~−52 kJ/mol per water is in line
with the SPC/E cohesive energy and a useful sanity check that the
charges, LJ, and SPME are wired correctly.
