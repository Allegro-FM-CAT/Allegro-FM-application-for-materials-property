import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase import units
from ase.neighborlist import build_neighbor_list, natural_cutoffs
from collections import defaultdict

# for VACF
def calculate_vdos(traj, timestep_fs, output_filename='vdos_analysis.png'):
    print("\n" + "="*60)
    print("\n DOS calculation from VACF...")
    print("-"*60)
    try:
        velocities = np.array([atoms.get_velocities() for atoms in traj])
    except Exception as e:
        print(f" Error getting velocities: {e}")
        return

    n_frames, n_atoms, _ = velocities.shape
    if n_frames < 2:
        print(" longer traj needed.")
        return

    # Calculate VACF using FFT for efficiency (Wiener-Khinchin theorem)
    vacf = np.zeros(n_frames)
    for i in range(n_atoms):
        for j in range(3): # x, y, z components
            vel_component = velocities[:, i, j]
            vel_padded = np.append(vel_component, np.zeros(n_frames)) # Zero-padding
            fft_vel = np.fft.fft(vel_padded)
            power_spectrum = np.abs(fft_vel)**2
            autocorr = np.fft.ifft(power_spectrum).real
            vacf += autocorr[:n_frames]
            
    # Normalize
    vacf /= (n_atoms * 3)
    if vacf[0] > 0:
        vacf /= vacf[0]

    # VDOS is FFT of VACF
    window = np.hanning(n_frames)
    vdos = np.abs(np.fft.fft(vacf * window))**2
    
    # Calculate corresponding frequencies in wavenumbers (cm^-1)
    timestep_s = timestep_fs * 1e-15
    freq_hz = np.fft.fftfreq(n_frames, d=timestep_s)
    wavenumbers = freq_hz / (units.C * 100) # Speed of light in cm/s

    mask = wavenumbers > 0
    
    plt.figure(figsize=(10, 6))
    plt.plot(wavenumbers[mask], vdos[mask], color='crimson')
    plt.title('Vibrational Density of States (VDOS)')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity (arb. units)')
    plt.xlim(0, 4500)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_filename, dpi=300)
    print(f" VDOS plot saved to '{output_filename}'")

def calculate_rdf(traj, rmax=10.0, nbins=200, pairs=None, output_filename='rdf_analysis.png'):
    print("\n" + "="*60)
    print("\n g(r) calculation...")
    print("-"*60) 
    
    symbols_list = traj[0].get_chemical_symbols()
    unique_symbols = sorted(list(set(symbols_list)))
    n_atoms = len(symbols_list)
    
    # If pairs list from --rdf flag is empty, compute all pairs
    if not pairs:
        pairs = []
        for i, s1 in enumerate(unique_symbols):
            for s2 in unique_symbols[i:]:
                pairs.append((s1, s2))
        print(f" No specific pairs provided. Computing RDF for all pairs: {pairs}")
    else:
        print(f" Computing RDF for specified pairs: {pairs}")

    # Check if system is periodic
    is_periodic = np.any(traj[0].get_pbc())
    
    # Determine if this is a small molecular system
    is_molecular = (n_atoms < 100 and not is_periodic)
    
    if is_periodic:
        print(" Detected periodic system")
        volume_avg = np.mean([atoms.get_volume() for atoms in traj])
    elif is_molecular:
        print(f" Detected small molecular system ({n_atoms} atoms)")
        print(" Note: For small molecules, consider using --pair_dist instead of --rdf")
        print("       RDF normalization may not be meaningful for few atoms")
        # Use a more reasonable effective volume for small molecules
        all_positions = np.vstack([atoms.get_positions() for atoms in traj])
        center = all_positions.mean(axis=0)
        max_extent = np.max(np.linalg.norm(all_positions - center, axis=1))
        # Use a box that's 2x the maximum extent
        volume_avg = (2 * max_extent) ** 3
        print(f" Using box volume: {volume_avg:.2f} Å³")
    else:
        print(" Detected non-periodic system (molecule/cluster)")
        all_positions = np.vstack([atoms.get_positions() for atoms in traj])
        max_extent = np.max(np.linalg.norm(all_positions - all_positions.mean(axis=0), axis=1))
        volume_avg = (4/3) * np.pi * (max_extent + rmax)**3
        print(f" Estimated effective volume: {volume_avg:.2f} Å³")

    hist_range = (0, rmax)
    g_r = {pair: np.zeros(nbins) for pair in pairs}
    total_frames = len(traj)
    
    for atoms in traj:
        symbols = np.array(atoms.get_chemical_symbols())
        
        for s1, s2 in pairs:
            indices1 = np.where(symbols == s1)[0]
            indices2 = np.where(symbols == s2)[0]
            
            if not len(indices1) or not len(indices2): 
                print(f" Warning: No atoms found for pair {s1}-{s2}")
                continue

            # Calculate distances manually to avoid broadcasting issues with get_distances
            # This is more robust for large systems
            for i in indices1:
                for j in indices2:
                    if s1 == s2 and i >= j:
                        continue  # Avoid double counting for same species
                    
                    if is_periodic:
                        dist = atoms.get_distance(i, j, mic=True)
                    else:
                        dist = atoms.get_distance(i, j, mic=False)
                    
                    if dist < rmax:
                        hist_idx = int(dist / rmax * nbins)
                        if 0 <= hist_idx < nbins:
                            g_r[(s1, s2)][hist_idx] += 1
            
    # Normalize
    plt.figure(figsize=(10, 6))
    
    # Create bin edges and radii
    edges = np.linspace(0, rmax, nbins + 1)
    radii = (edges[:-1] + edges[1:]) / 2.0
    dr = edges[1] - edges[0]
    
    for s1, s2 in pairs:
        num_atoms1 = symbols_list.count(s1)
        num_atoms2 = symbols_list.count(s2)
        if not num_atoms1 or not num_atoms2: continue

        # Shell volumes: 4π r² dr
        shell_volumes = 4.0 * np.pi * radii**2 * dr
        
        # Avoid division by zero for very small radii
        # Set a minimum shell volume threshold
        shell_volumes = np.maximum(shell_volumes, 1e-10)
        
        if s1 == s2:
            norm_factor = (num_atoms1 * (num_atoms1 - 1)) / (2 * volume_avg)
        else:
            norm_factor = (num_atoms1 * num_atoms2) / volume_avg

        # Avoid division by zero in normalization
        if norm_factor == 0:
            print(f" Warning: Zero normalization factor for {s1}-{s2}")
            continue
        
        avg_hist = g_r[(s1, s2)] / total_frames
        final_gr = avg_hist / (shell_volumes * norm_factor)
        
        # For non-periodic systems, mask out very small distances where g(r) is ill-defined
        if not is_periodic:
            mask = radii > 0.1  # Ignore distances below 0.1 Å
            plt.plot(radii[mask], final_gr[mask], label=f'{s1}-{s2}')
        else:
            plt.plot(radii, final_gr, label=f'{s1}-{s2}')

    plt.title('RDF g(r)')
    plt.xlabel('Distance r (Å)')
    plt.ylabel('g(r)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_filename, dpi=300)
    print(f" RDF plot saved to '{output_filename}'")


def analyze_pair_distances(traj, rmax=5.0, pairs=None, output_filename='pair_distance_distribution.png'):
    """
    Calculates and plots the distribution of interatomic distances.
    Better for small molecular systems than RDF.
    """
    print("\n" + "="*60)
    print("\n Pair distance distribution analysis...")
    print("-"*60)
    
    symbols_list = traj[0].get_chemical_symbols()
    unique_symbols = sorted(list(set(symbols_list)))
    
    # If pairs list is empty, compute all pairs
    if not pairs:
        pairs = []
        for i, s1 in enumerate(unique_symbols):
            for s2 in unique_symbols[i:]:
                pairs.append((s1, s2))
        print(f" Analyzing all pairs: {pairs}")
    else:
        print(f" Analyzing specified pairs: {pairs}")
    
    # Collect distances for each pair type
    pair_distances = {pair: [] for pair in pairs}
    
    for atoms in traj:
        symbols = np.array(atoms.get_chemical_symbols())
        
        for s1, s2 in pairs:
            indices1 = np.where(symbols == s1)[0]
            indices2 = np.where(symbols == s2)[0]
            
            if not len(indices1) or not len(indices2):
                continue
            
            for i in indices1:
                for j in indices2:
                    if s1 == s2 and i >= j:
                        continue  # Avoid double counting for same species
                    dist = atoms.get_distance(i, j, mic=False)
                    if dist <= rmax:
                        pair_distances[(s1, s2)].append(dist)
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = ['crimson', 'navy', 'forestgreen', 'darkorange', 'purple', 'brown']
    
    for idx, (pair, distances) in enumerate(pair_distances.items()):
        if not distances:
            print(f" Warning: No distances found for {pair[0]}-{pair[1]}")
            continue
        
        color = colors[idx % len(colors)]
        plt.hist(distances, bins=100, range=(0, rmax), alpha=0.7, 
                label=f'{pair[0]}-{pair[1]} (n={len(distances)})', 
                color=color, edgecolor='black', linewidth=0.5)
        
        # Print statistics
        print(f" {pair[0]}-{pair[1]}: mean={np.mean(distances):.3f} Å, "
              f"std={np.std(distances):.3f} Å, min={np.min(distances):.3f} Å, "
              f"max={np.max(distances):.3f} Å")
    
    plt.title('Pair Distance Distribution')
    plt.xlabel('Distance (Å)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_filename, dpi=300)
    print(f" Pair distance plot saved to '{output_filename}'")


def analyze_geometry(traj, bond_pairs=None, angle_triplets=None, cutoff_multiplier=1.2):
    print("\n" + "="*60)
    print("\n Bond and Angle analysis...")
    print("-"*60)

    if not bond_pairs and not angle_triplets:
        print(" No bond or angle pairs specified. Skipping geometry analysis.")
        return

    if bond_pairs:
        print(f" Analyzing bonds: {bond_pairs}")
    if angle_triplets:
        print(f" Analyzing angles: {angle_triplets}")

    bond_data = defaultdict(list)
    angle_data = defaultdict(list)
    
    for atoms in traj:
        symbols = np.array(atoms.get_chemical_symbols())
        cutoffs = natural_cutoffs(atoms, mult=cutoff_multiplier)
        nl = build_neighbor_list(atoms, cutoffs=cutoffs, self_interaction=False, bothways=True)
        
        # Analyze bonds
        if bond_pairs:
            for i in range(len(atoms)):
                s1 = symbols[i]
                indices, offsets = nl.get_neighbors(i)
                for j, offset in zip(indices, offsets):
                    if i >= j: continue # Avoid double counting
                    s2 = symbols[j]
                    if tuple(sorted((s1, s2))) in bond_pairs:
                        dist = np.linalg.norm(atoms.positions[j] + np.dot(offset, atoms.cell) - atoms.positions[i])
                        bond_data[tuple(sorted((s1, s2)))].append(dist)
        
        # Analyze angles
        if angle_triplets:
             for i in range(len(atoms)): # Center atom
                s_center = symbols[i]
                neighbors_i, _ = nl.get_neighbors(i)
                for j_idx, j in enumerate(neighbors_i):
                    s_j = symbols[j]
                    for k in neighbors_i[j_idx+1:]:
                        s_k = symbols[k]
                        triplet = tuple(sorted((s_j, s_k))) + (s_center,)
                        if triplet in angle_triplets:
                             angle_data[triplet].append(atoms.get_angle(j, i, k))

    # Plotting
    if bond_pairs:
        for pair, data in bond_data.items():
            if not data:
                print(f" Warning: No bond data found for {pair[0]}-{pair[1]}")
                continue
            plt.figure(figsize=(8, 5))
            plt.hist(data, bins=100, alpha=0.8)
            plt.title(f'Bond Length Distribution: {pair[0]}-{pair[1]}')
            plt.xlabel('Bond Length (Å)')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(f'bonds_{pair[0]}-{pair[1]}.png', dpi=300)
            print(f" Bond plot saved to 'bonds_{pair[0]}-{pair[1]}.png'")
    
    if angle_triplets:
        for triplet, data in angle_data.items():
            if not data:
                print(f" Warning: No angle data found for {triplet[0]}-{triplet[2]}-{triplet[1]}")
                continue
            plt.figure(figsize=(8, 5))
            plt.hist(data, bins=100, alpha=0.8)
            plt.title(f'Angle Distribution: {triplet[0]}-{triplet[2]}-{triplet[1]}')
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(f'angles_{triplet[0]}-{triplet[2]}-{triplet[1]}.png', dpi=300)
            print(f" Angle plot saved to 'angles_{triplet[0]}-{triplet[2]}-{triplet[1]}.png'")

def main():
    parser = argparse.ArgumentParser(
        description="General analysis script for ASE MD trajectories."
    )
    parser.add_argument('traj_file', type=str, help="Path to the trajectory file.")
    parser.add_argument('--timestep', type=float, default=1.0, help="MD timestep in fs.")
    parser.add_argument('--rdf', nargs='*', help="Calculate RDF for specific pairs (e.g., Si-O O-O). If no pairs are given, calculates for all pairs.")
    parser.add_argument('--rdf_rmax', type=float, default=10.0, help="Maximum distance for RDF calculation (Å).")
    parser.add_argument('--rdf_nbins', type=int, default=200, help="Number of bins for RDF calculation.")
    parser.add_argument('--pair_dist', nargs='*', help="Calculate simple pair distance distributions (better for molecules). Specify pairs or leave empty for all.")
    parser.add_argument('--pair_dist_rmax', type=float, default=5.0, help="Maximum distance for pair distance analysis (Å).")
    parser.add_argument('--bonds', nargs='+', help="Analyze bond lengths for specific pairs (e.g., H-O).")
    parser.add_argument('--angles', nargs='+', help="Analyze bond angles for specific triplets (e.g., H-O-H).")
    parser.add_argument('--cutoff_multiplier', type=float, default=1.2, help="Multiplier for ASE's natural_cutoffs for bond finding.")

    args = parser.parse_args()

    # --- Load Trajectory ---
    print(f"\n Loading trajectory from: {args.traj_file}")
    try:
        traj = read(args.traj_file, index=':')
        print(f" Successfully loaded {len(traj)} frames.")
    except Exception as e:
        print(f" Error: Could not read the trajectory file. {e}")
        return

    # --- Parse Arguments ---
    bond_pairs = None
    if args.bonds:
        bond_pairs = [tuple(sorted(p.split('-'))) for p in args.bonds]
    
    angle_triplets = None
    if args.angles:
        parsed_triplets = []
        for t in args.angles:
            parts = t.split('-')
            if len(parts) != 3:
                print(f" Warning: Invalid angle format '{t}'. Expected format: A-B-C")
                continue
            # Store as (sorted outer atoms, center atom)
            parsed_triplets.append(tuple(sorted((parts[0], parts[2]))) + (parts[1],))
        angle_triplets = parsed_triplets if parsed_triplets else None

    # --- Run Analyses ---
    if args.rdf is not None:
        rdf_pairs = None
        if args.rdf:  # If specific pairs were provided
            rdf_pairs = [tuple(sorted(p.split('-'))) for p in args.rdf]
        calculate_rdf(traj, rmax=args.rdf_rmax, nbins=args.rdf_nbins, pairs=rdf_pairs)

    if args.pair_dist is not None:
        pair_dist_pairs = None
        if args.pair_dist:  # If specific pairs were provided
            pair_dist_pairs = [tuple(sorted(p.split('-'))) for p in args.pair_dist]
        analyze_pair_distances(traj, rmax=args.pair_dist_rmax, pairs=pair_dist_pairs)

    # FIX: Changed condition to check for None instead of truthiness
    if bond_pairs is not None or angle_triplets is not None:
        analyze_geometry(traj, bond_pairs=bond_pairs, angle_triplets=angle_triplets, cutoff_multiplier=args.cutoff_multiplier)
        
    calculate_vdos(traj, args.timestep)

    print("\n" + "="*60)
    print("\n Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()