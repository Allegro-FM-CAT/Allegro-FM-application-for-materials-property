from ase.build import bulk
import numpy as np
import matplotlib.pyplot as plt
from nequip.ase import NequIPCalculator
import torch

# Initialize the nequip calculator
calculator = NequIPCalculator.from_compiled_model(
    compile_path="nequip_models_rc52/s64_t128.nequip.pth",
    chemical_symbols=["Ni"],
    device="cuda" if torch.cuda.is_available() else "cpu",
)  # use GPUs if available

# Range of scaling factors for lattice constant
scaling_factors = np.linspace(0.95, 1.05, 10)
volumes = []
energies = []

# Loop through scaling factors, calculate energy, and collect volumes and energies
for scale in scaling_factors:

    # Generate the cubic silicon structure with 216 atoms
    scaled_si = bulk("Ni", crystalstructure="fcc", a=3.52 * scale, cubic=True)
    scaled_si *= (3, 3, 3)  # Make a supercell (3x3x3) to get 216 atoms
    scaled_si.calc = calculator

    volume = scaled_si.get_volume()
    energy = scaled_si.get_potential_energy()
    volumes.append(volume)
    energies.append(energy)

# Plot the energy-volume curve
plt.tight_layout()
plt.figure(figsize=(8, 6))
plt.plot(volumes, energies, marker="o", label="E-V Curve")
plt.xlabel("Volume (Å³)", fontsize=14)
plt.ylabel("Energy (eV)", fontsize=14)
plt.title("Energy-Volume Curve for FCC Ni", fontsize=16)
plt.legend(fontsize=12)
plt.savefig("energy_volume_Ni.png")
