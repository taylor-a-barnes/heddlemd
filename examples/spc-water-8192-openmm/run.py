#!/usr/bin/env python3
"""OpenMM analogue of ../spc-water-8192 for cross-engine benchmarking.

Reads the same extended-XYZ initial state (`../spc-water-8192/water.in.xyz`),
builds the equivalent SPC/E + SPME + rigid-SETTLE system in OpenMM, and runs
the same three-phase schedule the HeddleMD TOML drives:

    min:   steepest-descent / L-BFGS energy minimization
    equil: 2,500 steps NVT at 298.15 K (5 ps @ dt = 2 fs)
    prod:  1,000 steps NPT at 298.15 K, 1.013e5 Pa (2 ps @ dt = 2 fs)

The wall-clock time of each phase is reported on stdout so it can be
compared against the `.timings` files written by `heddlemd run` on the
same machine.

Physics-level differences vs. the HeddleMD example (these affect the
microscopic trajectory but not the *amount of work* per step, which is
what the benchmark measures):

  - Thermostat: HeddleMD uses CSVR (Bussi-Donadio-Parrinello, global
    velocity rescaling). OpenMM does not ship CSVR; we use
    `LangevinMiddleIntegrator`, the standard OpenMM NVT integrator.
    Both sample the canonical distribution.
  - Barostat: HeddleMD uses c-rescale (Bernetti-Bussi, continuous
    stochastic). OpenMM ships `MonteCarloBarostat` (discrete MC volume
    moves). Both sample the isobaric-isothermal ensemble.
  - Constraints: HeddleMD uses SHAKE on three pair constraints per
    water. OpenMM auto-detects the rigid-water shape from the three
    constraints declared below and uses SETTLE (analytic, faster).
  - Precision: HeddleMD is f32 end-to-end. OpenMM defaults to "mixed"
    here (forces f32, accumulators f64) — the closest analogue is the
    "single" precision mode, switchable via `--single`.

Numeric constants are taken from `../spc-water-8192/water.in.toml`
verbatim (masses in kg, charges in C, LJ in m / J) and converted at
runtime, so any rounding the HeddleMD TOML carries is mirrored here.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit


HERE = Path(__file__).resolve().parent
DEFAULT_INIT = HERE.parent / "spc-water-8192" / "water.in.xyz"

# --- SPC/E numeric constants, copied verbatim from water.in.toml ---
# masses (kg)
M_O_KG = 2.6566e-26
M_H_KG = 1.6735e-27
# charges (C)
Q_O_C = -1.3585e-19
Q_H_C = +6.7925e-20
# LJ (O-O only; H-H and O-H carry negligible epsilon in the TOML — set to 0 here)
SIGMA_OO_M = 3.166e-10
EPSILON_OO_J = 1.080e-21        # per pair, in joules
R_CUT_M = 1.0e-9
# rigid SPC/E geometry
R_OH_M = 1.0e-10
R_HH_M = 1.633e-10
# SPME
PME_ALPHA_PER_M = 3.5e9
PME_GRID = (48, 48, 96)
# integrator / thermostat / barostat
DT_FS = 2.0
T_KELVIN = 298.15
TAU_THERMOSTAT_S = 1.0e-13      # 100 fs; gamma = 1/tau for Langevin
P_PA = 1.01325e5

# Avogadro for J -> kJ/mol (OpenMM's natural energy unit).
N_A = 6.02214076e23


def kj_per_mol(joules_per_pair):
    """Convert pair energy in J to kJ/mol (OpenMM's energy unit)."""
    return joules_per_pair * N_A * 1.0e-3


def read_extxyz(path):
    """Parse a HeddleMD extended-XYZ init file.

    Returns:
        species:    list[str], length N
        positions:  ndarray (N, 3), nanometers
        box_nm:     ndarray (3,), orthorhombic box lengths in nm
    """
    with open(path) as f:
        n_atoms = int(f.readline().strip())
        header = f.readline()
        lattice_str = header.split('Lattice="', 1)[1].split('"', 1)[0]
        lattice = [float(x) for x in lattice_str.split()]
        ax, ay, az, bx, by, bz, cx, cy, cz = lattice
        if any(abs(v) > 1e-20 for v in (ay, az, bx, bz, cx, cy)):
            raise ValueError("Non-orthorhombic lattice not supported")
        box_nm = np.array([ax, by, cz], dtype=np.float64) * 1.0e9
        species = []
        positions = np.zeros((n_atoms, 3), dtype=np.float64)
        for i in range(n_atoms):
            tok = f.readline().split()
            species.append(tok[0])
            positions[i, 0] = float(tok[1]) * 1.0e9
            positions[i, 1] = float(tok[2]) * 1.0e9
            positions[i, 2] = float(tok[3]) * 1.0e9
    return species, positions, box_nm


def build_system(species, box_nm):
    """Build the OpenMM System for rigid SPC/E water with SPME."""
    n_atoms = len(species)
    if n_atoms % 3 != 0:
        raise ValueError(f"Atom count {n_atoms} not a multiple of 3 (O,H,H)")
    n_waters = n_atoms // 3

    system = mm.System()
    box_vecs = [
        mm.Vec3(box_nm[0], 0.0, 0.0) * unit.nanometer,
        mm.Vec3(0.0, box_nm[1], 0.0) * unit.nanometer,
        mm.Vec3(0.0, 0.0, box_nm[2]) * unit.nanometer,
    ]
    system.setDefaultPeriodicBoxVectors(*box_vecs)

    m_o = M_O_KG * unit.kilogram / unit.item
    m_h = M_H_KG * unit.kilogram / unit.item
    for s in species:
        if s == "O":
            system.addParticle(m_o)
        elif s == "H":
            system.addParticle(m_h)
        else:
            raise ValueError(f"Unknown species {s!r}")

    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(R_CUT_M * unit.meter)
    nb.setUseDispersionCorrection(False)
    nb.setUseSwitchingFunction(False)
    nb.setPMEParameters(PME_ALPHA_PER_M / 1.0e9, *PME_GRID)

    sigma_oo = SIGMA_OO_M * unit.meter
    epsilon_oo = kj_per_mol(EPSILON_OO_J) * unit.kilojoule_per_mole
    # OpenMM has no built-in C <-> e conversion factor, so divide manually.
    e_in_coulombs = 1.602176634e-19
    q_o = (Q_O_C / e_in_coulombs) * unit.elementary_charge
    q_h = (Q_H_C / e_in_coulombs) * unit.elementary_charge
    zero_sigma = 0.1 * unit.nanometer       # arbitrary; epsilon=0 zeroes the term
    zero_epsilon = 0.0 * unit.kilojoule_per_mole

    for s in species:
        if s == "O":
            nb.addParticle(q_o, sigma_oo, epsilon_oo)
        else:
            nb.addParticle(q_h, zero_sigma, zero_epsilon)

    # Exclude intramolecular Coulomb + LJ (rigid water has no 1-2 or 1-3
    # nonbonded contribution). Matches the HeddleMD setup, which treats
    # the three atoms of a constraint group as exclusion-bonded.
    for m_idx in range(n_waters):
        o, h1, h2 = 3 * m_idx, 3 * m_idx + 1, 3 * m_idx + 2
        nb.addException(o, h1, 0.0, zero_sigma, zero_epsilon)
        nb.addException(o, h2, 0.0, zero_sigma, zero_epsilon)
        nb.addException(h1, h2, 0.0, zero_sigma, zero_epsilon)

    system.addForce(nb)

    # Rigid SPC/E: three distance constraints per molecule. OpenMM detects
    # the water-shape topology and dispatches to SETTLE automatically.
    r_oh = R_OH_M * unit.meter
    r_hh = R_HH_M * unit.meter
    for m_idx in range(n_waters):
        o, h1, h2 = 3 * m_idx, 3 * m_idx + 1, 3 * m_idx + 2
        system.addConstraint(o, h1, r_oh)
        system.addConstraint(o, h2, r_oh)
        system.addConstraint(h1, h2, r_hh)

    return system, n_waters


def make_context(system, platform, precision, integrator):
    properties = {}
    if platform.getName() == "CUDA":
        properties["Precision"] = precision
        properties["DeterministicForces"] = "true"
    return mm.Context(system, integrator, platform, properties)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", type=Path, default=DEFAULT_INIT,
                        help="Path to extended-XYZ initial state.")
    parser.add_argument("--platform", default="CUDA",
                        choices=["CUDA", "CPU", "Reference"])
    parser.add_argument("--precision", default="mixed",
                        choices=["single", "mixed", "double"],
                        help="CUDA platform precision (ignored on CPU/Reference).")
    parser.add_argument("--equil-steps", type=int, default=2500)
    parser.add_argument("--prod-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed for initial Maxwell-Boltzmann velocity sampling.")
    args = parser.parse_args()

    species, positions_nm, box_nm = read_extxyz(args.init)
    n_atoms = len(species)
    print(f"loaded {n_atoms} atoms ({n_atoms // 3} waters) from {args.init}")
    print(f"box: {box_nm[0]:.3f} x {box_nm[1]:.3f} x {box_nm[2]:.3f} nm")

    system, n_waters = build_system(species, box_nm)
    positions = positions_nm * unit.nanometer

    platform = mm.Platform.getPlatformByName(args.platform)
    print(f"platform: {platform.getName()}"
          + (f" (precision={args.precision})" if args.platform == "CUDA" else ""))

    dt = DT_FS * unit.femtoseconds
    temperature = T_KELVIN * unit.kelvin
    friction = (1.0 / TAU_THERMOSTAT_S) / unit.second   # ~10 ps^-1

    # --- Phase 1: minimization ---
    integrator_min = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = make_context(system, platform, args.precision, integrator_min)
    context.setPositions(positions)

    t0 = time.perf_counter()
    mm.LocalEnergyMinimizer.minimize(
        context,
        tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=500,
    )
    t_min = time.perf_counter() - t0
    state = context.getState(getPositions=True, getEnergy=True)
    pe_min = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    min_positions = state.getPositions()
    print(f"[min]   wall = {t_min:7.3f} s   "
          f"U = {pe_min:.6e} kJ/mol")
    del context, integrator_min

    # --- Phase 2: NVT equilibration ---
    integrator_equil = mm.LangevinMiddleIntegrator(temperature, friction, dt)
    integrator_equil.setRandomNumberSeed(11)
    context = make_context(system, platform, args.precision, integrator_equil)
    context.setPositions(min_positions)
    context.setVelocitiesToTemperature(temperature, args.seed)

    # Warm-up: first kernel launches include JIT and cuFFT plan creation; pulling
    # them out of the timed region produces a fair steady-state ms/step number.
    integrator_equil.step(10)
    t0 = time.perf_counter()
    integrator_equil.step(args.equil_steps)
    state = context.getState(getPositions=True, getVelocities=True,
                             enforcePeriodicBox=False)
    t_equil = time.perf_counter() - t0
    print(f"[equil] wall = {t_equil:7.3f} s   "
          f"steps = {args.equil_steps:5d}   "
          f"{1e3 * t_equil / args.equil_steps:6.2f} ms/step")
    equil_positions = state.getPositions()
    equil_velocities = state.getVelocities()
    equil_box = state.getPeriodicBoxVectors()
    del context, integrator_equil

    # --- Phase 3: NPT production ---
    barostat = mm.MonteCarloBarostat(P_PA * unit.pascal, temperature, 25)
    barostat.setRandomNumberSeed(13)
    system.addForce(barostat)
    integrator_prod = mm.LangevinMiddleIntegrator(temperature, friction, dt)
    integrator_prod.setRandomNumberSeed(12)
    context = make_context(system, platform, args.precision, integrator_prod)
    context.setPeriodicBoxVectors(*equil_box)
    context.setPositions(equil_positions)
    context.setVelocities(equil_velocities)

    integrator_prod.step(10)
    t0 = time.perf_counter()
    integrator_prod.step(args.prod_steps)
    state = context.getState(getEnergy=True, getPositions=False)
    t_prod = time.perf_counter() - t0
    print(f"[prod]  wall = {t_prod:7.3f} s   "
          f"steps = {args.prod_steps:5d}   "
          f"{1e3 * t_prod / args.prod_steps:6.2f} ms/step")

    print()
    print(f"total wall (min + equil + prod) = {t_min + t_equil + t_prod:.3f} s")


if __name__ == "__main__":
    main()
