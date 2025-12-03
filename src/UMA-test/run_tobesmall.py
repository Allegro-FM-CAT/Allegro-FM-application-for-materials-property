#!/usr/bin/env python3
"""Runner for the `tobe-small.xyz` molecular test using UMA model.

Defaults target the `tobe-small.xyz` file in the workspace root and the model
`uma-s-1.pt`. The script does a quick relax followed by a short MD and writes
out trajectories and the final structure.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description='Run relax+MD on tobe-small.xyz using UMA')
    parser.add_argument('--xyz', type=str, default='tobe-small.xyz', help='Path to xyz file')
    parser.add_argument('--model', type=str, default='./uma-s-1.pt', help='Path to UMA model file')
    parser.add_argument('--device', type=str, default='cpu', help='Device for model (cpu or cuda)')
    parser.add_argument('--relax-steps', type=int, default=100, help='Max relax steps')
    parser.add_argument('--fmax', type=float, default=0.05, help='Force tolerance (eV/Å)')
    parser.add_argument('--md-steps', type=int, default=100, help='Number of MD steps')
    parser.add_argument('--timestep', type=float, default=1.0, help='Timestep in fs')
    parser.add_argument('--temperature', type=float, default=300.0, help='MD temperature in K')
    args = parser.parse_args()

    try:
        from ase.io import read, write
        from ase.optimize import LBFGS
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet
        from ase.md.langevin import Langevin
        from ase import units
        from ase.io import Trajectory
    except Exception as e:
        print('Error: ASE not available. Install ASE to run this script.')
        print('Detail:', e)
        sys.exit(2)

    try:
        from fairchem.core.units.mlip_unit import load_predict_unit
        from fairchem.core import FAIRChemCalculator
    except Exception as e:
        print('Error: UMA/fairchem imports failed. Ensure UMA package and model are available.')
        print('Detail:', e)
        sys.exit(2)

    xyz_path = args.xyz
    model_path = args.model

    if not os.path.exists(xyz_path):
        print(f"XYZ file not found at {xyz_path}. Provide correct path or place the file there.")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"UMA model not found at {model_path}. Provide --model path.")
        sys.exit(1)

    print(f"Reading structure from: {xyz_path}")
    atoms = read(xyz_path)

    print('Loading UMA model...')
    pred = load_predict_unit(path=model_path, device=args.device)
    # molecular task
    calc = FAIRChemCalculator(pred, task_name='omol')
    atoms.calc = calc

    print('\n' + '=' * 60)
    print('Initial evaluation')
    print('-' * 60)
    print('Energy (eV):', atoms.get_potential_energy())
    print('Max force (eV/Å):', abs(atoms.get_forces()).max())

    print('\n' + '=' * 60)
    print('Relaxing')
    opt = LBFGS(atoms, trajectory='tobesmall_opt.traj')
    opt.run(fmax=args.fmax, steps=args.relax_steps)
    print('Relaxed energy:', atoms.get_potential_energy())

    print('\n' + '=' * 60)
    print('Running MD')
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    dyn = VelocityVerlet(atoms, args.timestep * units.fs)
    traj = Trajectory('tobesmall_md.traj', 'w', atoms)
    dyn.attach(traj.write, interval=10)

    def status():
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * len(atoms))
        print(f"Step: {dyn.nsteps:5d}  E_pot: {epot:10.6f} eV  T: {temp:6.1f} K")

    dyn.attach(status, interval=10)
    dyn.run(args.md_steps)

    write('tobesmall_final.xyz', atoms)
    print('Wrote: tobesmall_opt.traj, tobesmall_md.traj, tobesmall_final.xyz')


if __name__ == '__main__':
    main()
