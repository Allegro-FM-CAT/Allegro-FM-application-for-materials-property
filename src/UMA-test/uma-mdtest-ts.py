from ase.build import fcc100, add_adsorbate, molecule
from ase.optimize import LBFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units
from ase.io import write, Trajectory
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator

# Setup initial structure water molecule
atoms = molecule("H2O")
atoms.info['charge'] = 0
atoms.info['spin_multiplicity'] = 1

# UMA model setup and calculator
print("Loading model...")
pred = load_predict_unit(path="./uma-s-1.pt", device="cpu")
calc = FAIRChemCalculator(pred, task_name="omol") # "omol" for molecules, "oc20" for materials/surfaces
atoms.calc = calc

# Initial Structure Evaluation
print("\n" + "="*60)
print("Initial Structure Evaluation")
print("-"*60)

E = atoms.get_potential_energy()  # eV
F = atoms.get_forces()            # eV/angstrom
print(f"Initial Energy: {E:.4f} eV")
print(f"Initial Forces (eV/Å):")
print(F)

# Geometry optimization before MD
print("\n" + "="*60)
print("Relaxing the structure")
print("-"*60)

opt = LBFGS(atoms, trajectory='optimization.traj')
opt.run(fmax=0.05, steps=100)

print(f"Optimized Energy: {atoms.get_potential_energy():.4f} eV")

# MD part
print("\n" + "="*60)
print("Starting MD Simulation")
print("-"*60)

# input parameters
temperature = 300  # K
timestep = 1.0     # fs
nsteps = 100       # number of MDsteps

# Initialize velocities according to Maxwell-Boltzmann distribution
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

# For MD integrator:
# +++++++++++++++++++++++++++++++++
# Option 1: Verlet (NVE)
dyn = VelocityVerlet(atoms, timestep * units.fs)

# +++++++++++++++++++++++++++++++++
# Option 2: Langevin thermostat (NVT)
# friction = 0.002  # friction coefficient in 1/fs
# dyn = Langevin(atoms, timestep * units.fs, temperature_K=temperature, 
#                friction=friction)

# Setup trajectory file
traj = Trajectory('md_trajectory.traj', 'w', atoms)
dyn.attach(traj.write, interval=10)

# Status printing function
def print_status():
    """Print current MD status"""
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    etot = epot + ekin
    temp = ekin / (1.5 * units.kB * len(atoms))
    
    print(f"Step: {dyn.nsteps:5d}  "
          f"E_pot: {epot:10.4f} eV  "
          f"E_kin: {ekin:8.4f} eV  "
          f"E_tot: {etot:10.4f} eV  "
          f"T: {temp:6.1f} K")

# printing status every ? steps
dyn.attach(print_status, interval=10)

# Print MD settings
print(f"Temperature: {temperature} K")
print(f"Timestep: {timestep} fs")
print(f"Total steps: {nsteps}")
print(f"Total time: {nsteps * timestep / 1000:.2f} ps")
print("-" * 60)

# Run MD
dyn.run(nsteps)

# Final MD print
print("\n" + "="*60)
print("MD finished.")
print("-"*60)

# Save final structure to xyz
write('final_structure.xyz', atoms)

print(f"Trajectory saved to: md_trajectory.traj")
print(f"Final structure saved to: final_structure.xyz")
print(f"Optimization trajectory saved to: optimization.traj")

#print("\nTo visualize trajectories:")
#print("  ase gui md_trajectory.traj")
#print("  ase gui optimization.traj")

# Simple analysis of trajectory
print("\n" + "="*60)
print("Trajectory Analysis")
print("-"*60)

from ase.io import read

traj_atoms = read('md_trajectory.traj', ':')
energies = [a.get_potential_energy() for a in traj_atoms]
temps = [a.get_kinetic_energy() / (1.5 * units.kB * len(a)) for a in traj_atoms]

print(f"Number of frames: {len(traj_atoms)}")
print(f"Average temperature: {sum(temps)/len(temps):.2f} K")
print(f"Average potential energy: {sum(energies)/len(energies):.4f} eV")
print(f"Energy range: {min(energies):.4f} to {max(energies):.4f} eV")

# To plot energy and temp vs MD steps
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

steps = [i * 10 for i in range(len(energies))]  # every 10 steps

ax1.plot(steps, energies, 'b-', linewidth=1)
ax1.set_xlabel('MDStep')
ax1.set_ylabel('Energy (eV)')
ax1.set_title('MD Energy over Time')
ax1.grid(True, alpha=0.3)

ax2.plot(steps, temps, 'r-', linewidth=1)
ax2.axhline(y=temperature, color='k', linestyle='--', label=f'Target: {temperature} K')
ax2.set_xlabel('MDStep')
ax2.set_ylabel('Temperature (K)')
ax2.set_title('MD Temperature over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('md_analysis.png', dpi=300)
print(f"\nPlot saved to: md_analysis.png")