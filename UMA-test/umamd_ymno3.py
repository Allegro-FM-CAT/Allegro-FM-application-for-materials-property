#!/usr/bin/env python3
"""Simple runner for YMnO3 using UMA model.

This script reads a POSCAR (default: 2509-mdapy/POSCAR), attaches the UMA
calculator (default model: ./uma-s-1.pt), performs a geometry relaxation and
then a short MD run, saving trajectories and final structure.

Usage examples:
  python run_ymno3.py
  python run_ymno3.py --poscar 2509-mdapy/POSCAR --model uma-s-1.pt --md-steps 200

The script tries to give helpful error messages if ASE or the UMA package is
missing. It intentionally keeps defaults conservative so you can test quickly.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Run simple relax + MD for YMnO3 using UMA model")
    parser.add_argument('--poscar', type=str, default='2509-mdapy/POSCAR', help='Path to POSCAR file')
    parser.add_argument('--model', type=str, default='./uma-s-1.pt', help='Path to UMA model file')
    parser.add_argument('--device', type=str, default='cpu', help='Device for the model (cpu or cuda)')
    parser.add_argument('--relax-steps', type=int, default=200, help='Max LBFGS relax steps')
    parser.add_argument('--fmax', type=float, default=0.05, help='Force tolerance (eV/Å) for relax')
    parser.add_argument('--md-steps', type=int, default=100, help='Number of MD steps')
    parser.add_argument('--timestep', type=float, default=1.0, help='MD timestep in fs')
    parser.add_argument('--temperature', type=float, default=300.0, help='MD target temperature (K)')
    parser.add_argument('--langevin', action='store_true', help='Use Langevin thermostat instead of NVE')
    parser.add_argument('--pairdist', action='store_true', help='Compute and plot pair distribution g(r) for last frame')
    parser.add_argument('--rmax', type=float, default=6.0, help='Max radius for g(r) in Å')
    parser.add_argument('--dr', type=float, default=0.05, help='Bin width for g(r) in Å')
    args = parser.parse_args()

    # Import heavy deps lazily to allow quick syntax-checks without installed packages
    try:
        from ase.io import read, write
        from ase.optimize import LBFGS
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet
        from ase.md.langevin import Langevin
        from ase import units
        from ase.io import Trajectory
    except Exception as e:
        print("Error: ASE imports failed. Make sure ASE is installed in your environment.")
        print("Detailed error:", e)
        sys.exit(2)

    try:
        from fairchem.core.units.mlip_unit import load_predict_unit
        from fairchem.core import FAIRChemCalculator
    except Exception as e:
        print("Error: UMA/fairchem imports failed. Ensure the UMA package and model files are available.")
        print("Detailed error:", e)
        sys.exit(2)

    poscar_path = args.poscar
    model_path = args.model

    if not os.path.exists(poscar_path):
        print(f"POSCAR not found at {poscar_path}. Please provide the correct path or place a POSCAR there.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"UMA model not found at {model_path}. Provide the model path via --model.")
        sys.exit(1)

    print(f"Reading structure from: {poscar_path}")
    atoms = read(poscar_path)

    print("Loading UMA model (this may take a little while)...")
    pred = load_predict_unit(path=model_path, device=args.device)
    # For bulk/materials set task_name to 'oc20' per umi example
    calc = FAIRChemCalculator(pred, task_name='oc20')
    atoms.calc = calc

    # Initial evaluation
    print('\n' + '=' * 60)
    print('Initial structure evaluation')
    print('-' * 60)
    E0 = atoms.get_potential_energy()
    F0 = atoms.get_forces()
    print(f"Initial energy: {E0:.6f} eV")
    print(f"Max initial force: {abs(F0).max():.6f} eV/Å")

    # Geometry optimization
    print('\n' + '=' * 60)
    print('Relaxing structure (LBFGS)')
    print('-' * 60)
    opt_traj = 'ymno3_optimization.traj'
    opt = LBFGS(atoms, trajectory=opt_traj)
    opt.run(fmax=args.fmax, steps=args.relax_steps)
    E_relaxed = atoms.get_potential_energy()
    print(f"Relaxed energy: {E_relaxed:.6f} eV")

    # MD setup
    print('\n' + '=' * 60)
    print('Starting short MD')
    print('-' * 60)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)

    if args.langevin:
        friction = 0.01
        dyn = Langevin(atoms, args.timestep * units.fs, temperature_K=args.temperature, friction=friction)
    else:
        dyn = VelocityVerlet(atoms, args.timestep * units.fs)

    md_traj = 'ymno3_md.traj'
    traj = Trajectory(md_traj, 'w', atoms)
    dyn.attach(traj.write, interval=10)

    def print_status():
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        etot = epot + ekin
        temp = ekin / (1.5 * units.kB * len(atoms))
        print(f"Step: {dyn.nsteps:5d}  E_pot: {epot:10.6f} eV  E_kin: {ekin:8.6f} eV  T: {temp:6.1f} K")

    dyn.attach(print_status, interval=10)

    print(f"MD steps: {args.md_steps}, timestep: {args.timestep} fs, target T: {args.temperature} K")
    dyn.run(args.md_steps)

    final_xyz = 'ymno3_final.xyz'
    write(final_xyz, atoms)

    print('\n' + '=' * 60)
    print('Finished. Files written:')
    for p in (opt_traj, md_traj, final_xyz):
        print(' -', p)

    # Pair distribution / RDF analysis on final frame
    if args.pairdist:
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from ase.neighborlist import neighbor_list
        except Exception as e:
            print('Error: numpy/matplotlib/ase.neighborlist required for pair distribution.')
            print('Detail:', e)
            sys.exit(1)

        print('\n' + '=' * 60)
        print('Computing pair distribution g(r) for final frame')
        print('-' * 60)

        atoms_final = atoms.copy()
        rmax = args.rmax
        dr = args.dr
        bins = np.arange(0.0, rmax + dr, dr)
        r = 0.5 * (bins[:-1] + bins[1:])

        # neighbor_list returns pairs within rmax (considering PBC)
        i_inds, j_inds, distances = neighbor_list('ijd', atoms_final, rmax)
        distances = np.array(distances)

        # Total g(r)
        hist, _ = np.histogram(distances, bins=bins)
        volume = atoms_final.get_volume()
        N = len(atoms_final)
        rho = N / volume
        shell_vol = 4.0 * np.pi * r**2 * dr
        # Normalize: divide by (rho * shell_vol * N_pairs_per_particle)
        g_r = hist / (rho * shell_vol * N)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(r, g_r, '-', lw=1.2)
        ax.set_xlabel('r (Å)')
        ax.set_ylabel('g(r)')
        ax.set_title('Pair distribution g(r) - total')
        ax.grid(True, alpha=0.3)
        fig.savefig('pair_distribution_ymo_total.png', dpi=300)
        print('Saved: pair_distribution_ymo_total.png')

        # Partial pairs for common species if present
        species = list(set(atoms_final.get_chemical_symbols()))
        targets = ['Y', 'Mn', 'O']
        available = [s for s in targets if s in species]
        if available:
            from collections import defaultdict
            partial_hist = defaultdict(lambda: np.zeros_like(r))
            symbols = atoms_final.get_chemical_symbols()
            # accumulate counts per bin for each unordered pair type
            for ia, ja, d in zip(i_inds, j_inds, distances):
                sa = symbols[ia]
                sb = symbols[ja]
                key = tuple(sorted((sa, sb)))
                idx = int((d - dr/2.0) // dr)
                if 0 <= idx < len(r):
                    partial_hist[key][idx] += 1

            fig, ax = plt.subplots(figsize=(7, 5))
            for key, hist_p in partial_hist.items():
                g_p = hist_p / (rho * shell_vol * N)
                name = f'{key[0]}-{key[1]}'
                ax.plot(r, g_p, label=name)
            ax.set_xlabel('r (Å)')
            ax.set_ylabel('g(r)')
            ax.set_title('Partial pair distributions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig('pair_distribution_ymo_partial.png', dpi=300)
            print('Saved: pair_distribution_ymo_partial.png')


if __name__ == '__main__':
    main()
