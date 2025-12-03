import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase import units

def analyze_geometry(traj):
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
    print("plot saved to 'water_bond_analysis.png'")

def analyze_pair_distances(traj):
    """
    for RDF
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
