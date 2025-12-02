# Allergo-FM-application-for-materials-property Proposal

Apply Allergo foundation model  and other popular foundation models for different materials properties prediction and analysis 

**Outline of project**:

**1. Introduction and Background**:
   - This project aims to use foundation model for materials research and molecular dynamics simulations.
   - We aims at exploring Allegro-FM, [Meta Uma universal model](https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/), [MACE model](https://github.com/ACEsuit/mace-foundations) and [PET-MAD model](https://github.com/lab-cosmo/pet-mad) to benchmark properties of Ni-based alloy system and tobermorite structure.

**2. Project goal**
   - Designing and characterizing Ni FCC structure and alloy system with Allegro-FM, including the effects of Ni percentage in alloy.
   - Validating the model's predictions against existing experimental or theoretical data for Ni-based alloy and tobermorite.
   - Testing lattice parameters, bond angle, bond length,elastic constant,g(r) and phonon properties.
   - Simulating phenomena at a larger scale, such as defect formation and favourable adsorption sites.
     
**3. Methodology**
   - Allegro-FM integration with [Atomic Simulation Environment](https://nequip.readthedocs.io/en/latest/integrations/ase.html): Use ASE package to build the single layer bulk and defective graphene.
   - Data Preparation for Fine-Tuning: Gather dataset of graphene or doped graphene structures and their corresponding energies and forces to use for fine-tuning.
   - Model Fine-tuning: Load pre-trained Allegro-FM with 89 elements, then fine-tune the model using the prepared graphene dataset to optimize for this particular material.
   - MD runs: Run [large-scale molecular dynamics simulations](https://www.lammps.org/#gsc.tab=0). Also simulate the effects of doping, such as changes in lattice structure or defect formation.
   - Result Analysis: Use ASE tools for preliminary analysis, then use tools like [phonopy](https://phonopy.github.io/phonopy/phonopy-module.html) or [mdapy](https://mdapy.readthedocs.io/en/latest/) for validation and characterization purpose. 

**4. Premilinary results**
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

- Lattice parameter prediction with MACE and UMA models for Tobermorite structure
