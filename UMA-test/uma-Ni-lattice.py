from ase.optimize import LBFGS
from dftd3.ase import DFTD3
from ase.visualize import view
from ase.eos import EquationOfState
from matplotlib import pyplot as plt
import numpy as np
from ase.build import bulk
from ase.io import read

atoms = read('Ni.xyz')

from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

# load UMA model locally
pred = load_predict_unit(path="./uma-s-1.pt", device="cpu")
calc = FAIRChemCalculator(pred, task_name="omat")  # omat is for inorganic materials
atoms.calc = calc

# relax structue
opt = LBFGS(atoms)
opt.run(0.02, 200)

# define lattice parameter
a0 = 3.52
scales = np.linspace(0.96, 1.04, 10)
energies,volumes = [],[]

for s in scales:
    a = 3.52*s
    test = bulk('Ni','fcc',a = a)
    test.calc = calc
    energies.append(test.get_potential_energy())
    volumes.append(test.get_volume())

eos = EquationOfState(volumes, energies)
v0,e0,B = eos.fit()

a_eq = (4.0*v0)**(1/3)
print('Equilibrum lattice (Å):',a_eq,"Bulk modulus (eV/Å^3):", B)
# convert eV/Å^3 to GPa
print('Equilibrum lattice (Å):',a_eq,"Bulk modulus (GPa):", B*160.217)

a_values = [(4.0*v)**(1/3) for v in volumes]

plt.figure(figsize=(12,8))
plt.plot(a_values,energies,'o-',label='UMA-Equilibrum lattice')
plt.xlabel('Lattice paramter a (Å)')
plt.ylabel('Potential Energy (eV)')
plt.legend()
plt.tight_layout()
plt.savefig('Ni_PE_vs_lattice.png', dpi=300)
plt.close()


