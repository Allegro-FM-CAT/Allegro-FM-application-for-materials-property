from ase import Atoms
from ase.phonons import Phonons
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
import numpy as np
import matplotlib.pyplot as plt

pred = load_predict_unit(path="./uma-s-1.pt", device="cpu")
calc = FAIRChemCalculator(pred, task_name="omat")

atoms_ni = bulk('Ni', 'fcc', a=3.52)
atoms_al = bulk('Al', 'fcc', a=4.05)

# Add a small displacement to break symmetry
atoms_ni.positions[1] += [0.01, 0, 0]
atoms_al.positions[1] += [0.01, 0, 0]

atoms_ni.calc = calc
atoms_al.calc = calc

# force and energy calculations
forces_ni = atoms_ni.get_forces()
forces_al = atoms_al.get_forces()
energy_ni = atoms_ni.get_potential_energy()
energy_al = atoms_al.get_potential_energy()

print(f"Ni forces (displaced): {forces_ni}")
print(f"Al forces (displaced): {forces_al}")
print(f"Ni energy: {energy_ni:.6f} eV")
print(f"Al energy: {energy_al:.6f} eV")

atoms_ni_test = bulk('Ni', 'fcc', a=3.52)
atoms_ni_test.positions[0] += [0.5, 0, 0]  # Larger displacement
atoms_ni_test.calc = calc

forces = atoms_ni_test.get_forces()
energy = atoms_ni_test.get_potential_energy()

print(f"Ni with large displacement:")
print(f"  Forces: {forces}")
print(f"  Energy: {energy:.6f} eV")
print(f"  Max force magnitude: {np.max(np.abs(forces))}")

# Test if stress is also zero
stress = atoms_ni_test.get_stress()
print(f"  Stress: {stress}")