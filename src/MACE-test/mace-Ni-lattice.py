from ase.build import fcc100, add_adsorbate, molecule
from ase.optimize import LBFGS
from dftd3.ase import DFTD3
from ase.visualize import view
from ase.filters import StrainFilter
from ase.io import read
from mace.calculators import  MACECalculator
from ase.build import bulk
from ase.eos import EquationOfState
import numpy as np
import matplotlib.pyplot as plt

# read the existing Ni structure
atoms = read('Ni.xyz')
a = 3.52                       # set up inital lattice parameter
atoms.set_cell(np.diag([a,a,a]), scale_atoms=True)
atoms.pbc = True

calc = MACECalculator(model_paths=["./mace_models/MACE-matpes-r2scan-omat-ft.model"], device="cpu")
atoms.calc = calc

E = atoms.get_potential_energy()
Fmax = (abs(atoms.get_forces())).max()
print("E (eV):", E, "Fmax (eV/Å):", Fmax)

# relax atoms (fixed cell)
opt = LBFGS(atoms)
opt.run(0.05, 100)
# relax cell
sf = StrainFilter(atoms)
LBFGS(sf, logfile=None).run(fmax=0.02)
print("lattice (Å):", atoms.cell.lengths()[0], "Energy/atom (eV):", atoms.get_potential_energy()/len(atoms))

# equilibrium lattice
scales = np.linspace(0.96, 1.04, 10)
volumes,energies = [],[]

for s in scales:
    a0 = 3.52*s
    test = bulk('Ni','fcc',a = a0)
    test.calc = calc
    energies.append(test.get_potential_energy())
    volumes.append(test.get_volume())

eos = EquationOfState(volumes, energies)
v0,e0,B = eos.fit()

a_eq = (4.0*v0)**(1/3)
print('Equilibrum lattice (Å):',a_eq,"Bulk modulus (eV/Å^3):", B)
print('Equilibrum lattice (Å):',a_eq,"Bulk modulus (GPa):", B*160.21)

# calculated based on geometry relationship with lattice parameter, FCC cubic, V = a^3, Vatom = Vcell/N_atom = a^3/4
a_values = [(4.0*v)**(1/3) for v in volumes]

# unify font size
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

plt.figure(figsize=(12,8))
plt.plot(a_values,energies,'o-',label='Equilibrum lattice')
plt.xlabel('Lattice paramter a (Å)')
plt.ylabel('Potential Energy (eV)')
plt.legend()
plt.tight_layout()
plt.savefig('Ni_PE_vs_lattice.png', dpi=300)
plt.close()

