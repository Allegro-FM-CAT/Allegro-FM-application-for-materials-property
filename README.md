# Allegro-FM: uMLIP Electronic Structure and Fine Tuning

**A materials simulation framework for benchmarking and fine-tuning of Foundation Models for Interatomic Potentials (Allegro-FM, UMA, MACE, PET-MAD).**

**Outline of project**:

**1. Introduction and Background**:

Atomistic simulation underpins modern materials science, providing a microscopic view of physical phenomena that govern macroscopic material behavior. Machine Learning Interatomic Potentials (MLIPs) have emerged as the necessary technical bridge to overcome this scalability challenge. MLIPs achieve accuracy comparable to DFT calculations while maintaining the computational efficiency inherent to classical molecular dynamics (MD) simulations.Foundation Models(FMs) are universal, pre-trained potentials capable of providing reliable, out-of-the-box predictions across a vast chemical space, thereby democratizing sophisticated atomistic modeling and significantly reducing the human effort previously required for developing system-specific potentials.
   - This project aims to use foundation model for materials research and molecular dynamics simulations.
   - We aims at exploring Allegro-FM, [Meta Uma universal model](https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/), [MACE model](https://github.com/ACEsuit/mace-foundations) and [PET-MAD model](https://github.com/lab-cosmo/pet-mad) to benchmark properties of Ni-based alloy system and tobermorite structure.

**2. Project goal**
   - Designing and characterizing Ni FCC structure and alloy system with Allegro-FM, including the effects of Ni percentage in alloy.
   - Validating the model's predictions against existing experimental or theoretical data for Ni-based alloy and tobermorite.
   - Testing lattice parameters, bond angle, bond length,elastic constant,g(r) and phonon properties.
   - Simulating phenomena at a larger scale, such as defect formation and favourable adsorption sites.
     
**3. Methodology**
   - Allegro-FM integration with [Atomic Simulation Environment](https://nequip.readthedocs.io/en/latest/integrations/ase.html): Use ASE package to build the FCC Ni structure and Ni-based alloy systems.
   - Data Preparation for Fine-Tuning: Gather dataset of Ni structures and their corresponding energies and forces to use for fine-tuning.
   - Model Fine-tuning: Load pre-trained Allegro-FM with 89 elements, then fine-tune the model using the prepared Ni-based alloy dataset to optimize for this particular material.
   - MD runs: Run [large-scale molecular dynamics simulations](https://www.lammps.org/#gsc.tab=0). Also simulate the effects of grain boudany, such as changes in lattice structure or stacking fault energy.
   - Result Analysis: Use ASE tools for preliminary analysis, then use tools like [phonopy](https://phonopy.github.io/phonopy/phonopy-module.html) or [mdapy](https://mdapy.readthedocs.io/en/latest/) for validation and characterization purpose. 

**4. Preliminary results**
- Lattice parameter prediction with MACE and UMA models for Ni FCC unit structure
<table>
  <tr>
    <td align="center">
      <img src="./MACE-test/Ni_PE_vs_lattice.png" alt="MACE Model" width="400"/>
      <br>
      <em>MACE Model Prediction</em>
    </td>
    <td align="center">
      <img src="./UMA-test/Ni_PE_vs_lattice.png" alt="UMA Model" width="400"/>
      <br>
      <em>UMA Model Prediction</em>
    </td>
  </tr>
</table>

- Potentoal energy variation with volume predicion with Nequip model for Ni FCC unit structure
![](./Nequip-test/energy_volume_Ni.png)

- Lattice parameter prediction with MACE and UMA models for Tobermorite structure
