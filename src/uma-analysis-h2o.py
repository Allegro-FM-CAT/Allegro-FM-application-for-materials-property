import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase import units

def analyze_geometry(traj):
    """
    Analyzes bond lengths and angles for a water molecule trajectory.
    Saves histograms of the distributions.
    Assumes the atoms are ordered O, H, H.
    """
    print("Analyzing geometry (bond lengths and angles)...")
    
    if len(traj[0]) != 3:
        print("Warning: Geometry analysis is tailored for a single water molecule.")
        return

    oh_lengths, hoh_angles = [], []
    for atoms in traj:
        # O-H bond lengths (indices 0-1 and 0-2)
        oh_lengths.append(atoms.get_distance(0, 1))
        oh_lengths.append(atoms.get_distance(0, 2))
        # H-O-H bond angle (indices 1-0-2)
        hoh_angles.append(atoms.get_angle(1, 0, 2))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bond Length Histogram
    ax1.hist(oh_lengths, bins=50, color='royalblue', alpha=0.7)
    ax1.set_title('O-H Bond Length Distribution')
    ax1.set_xlabel('Bond Length (Å)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Bond Angle Histogram
    ax2.hist(hoh_angles, bins=50, color='seagreen', alpha=0.7)
    ax2.set_title('H-O-H Bond Angle Distribution')
    ax2.set_xlabel('Bond Angle (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('water_bond_analysis.png', dpi=300)
    print("-> Geometry analysis plot saved to 'water_bond_analysis.png'")

def analyze_pair_distances(traj):
    """
    Calculates and plots the distribution of all interatomic distances.
    For a single molecule, this shows intramolecular distances.
    """
    print("\nAnalyzing pair distance distribution...")
    
    all_distances = []
    for atoms in traj:
        # Get all unique pairwise distances
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                all_distances.append(atoms.get_distance(i, j))
    
    plt.figure(figsize=(8, 5))
    plt.hist(all_distances, bins=100, range=(0, 3), color='darkorange', alpha=0.8)
    plt.title('Distribution of Intramolecular Distances')
    plt.xlabel('Distance (Å)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('water-gr.png', dpi=300)
    print("-> Pair distance plot saved to 'water-gr.png'")

def calculate_vdos(traj, timestep_fs):
    """
    Calculates the Vibrational Density of States (VDOS) from the
    Velocity Autocorrelation Function (VACF).
    """
    print("\nCalculating Vibrational Density of States (VDOS)...")
    
    try:
        velocities = np.array([atoms.get_velocities() for atoms in traj])
    except Exception as e:
        print(f"Error getting velocities: {e}")
        print("Please ensure velocities are saved in the trajectory file.")
        return

    n_frames, n_atoms, _ = velocities.shape
    
    # Calculate VACF using FFT for efficiency
    # The Wiener-Khinchin theorem states that the power spectral density
    # is the Fourier transform of the autocorrelation.
    vacf = np.zeros(n_frames)
    for i in range(n_atoms):
        for j in range(3): # x, y, z components
            vel_component = velocities[:, i, j]
            # Zero-pad for FFT
            vel_padded = np.append(vel_component, np.zeros(n_frames - 1))
            fft_vel = np.fft.fft(vel_padded)
            power_spectrum = np.abs(fft_vel)**2
            autocorr = np.fft.ifft(power_spectrum).real
            vacf += autocorr[:n_frames]
            
    # Normalize
    vacf /= (n_atoms * 3)
    if vacf[0] > 0:
        vacf /= vacf[0]

    # VDOS is the Fourier transform of the VACF
    # Apply a window function to reduce spectral leakage
    window = np.hanning(n_frames)
    vdos = np.abs(np.fft.fft(vacf * window))**2
    
    # Calculate corresponding frequencies in wavenumbers (cm^-1)
    timestep_s = timestep_fs * 1e-15
    freq_hz = np.fft.fftfreq(n_frames, d=timestep_s)
    wavenumbers = freq_hz / units.C * 1e-2 # units.C is speed of light in m/s

    # Keep only positive frequencies for plotting
    mask = wavenumbers > 0
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(wavenumbers[mask], vdos[mask], color='crimson')
    plt.title('Vibrational Density of States (VDOS)')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity (arb. units)')
    plt.xlim(0, 4500) # Typical range for molecular vibrations
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('water-vdos.png', dpi=300)
    print("-> VDOS plot saved to 'water-vdos.png'")

def main():
    parser = argparse.ArgumentParser(
        description="""Analysis script for ASE MD trajectories.
        Calculates geometric properties and vibrational density of states."""
    )
    parser.add_argument(
        'traj_file', 
        type=str, 
        help="Path to the ASE trajectory file (e.g., 'md_trajectory.traj')."
    )
    parser.add_argument(
        '--timestep', 
        type=float, 
        default=1.0, 
        help="MD timestep in femtoseconds (fs). Default: 1.0."
    )
    args = parser.parse_args()

    print(f"Loading trajectory from: {args.traj_file}")
    try:
        traj = read(args.traj_file, index=':')
        print(f"Successfully loaded {len(traj)} frames.")
    except Exception as e:
        print(f"Error: Could not read the trajectory file. {e}")
        return
        
    # --- Run Analyses ---
    analyze_geometry(traj)
    analyze_pair_distances(traj)
    calculate_vdos(traj, args.timestep)

    print("\n Analysis complete.")

if __name__ == "__main__":
    main()