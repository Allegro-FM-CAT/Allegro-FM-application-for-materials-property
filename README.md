# Allergo-FM-application-for-materials-property

Apply Allergo foundation model for different materials properties analysis Proposal

**Outline of project**:

**1. Introduction and Background**:
   - This project aims to use foundation model for materials research and molecular dynamics.
   - We aims at exploring Allegro-FM foundation model and also [Meta Uma universal model](https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/).

**2. Project goal**
   - Designing and characterizing single-Layer graphene with Allegro-FM, including the effects of target doping and point defects.
   - Validating the model's predictions against existing experimental or theoretical data for 2D graphene.
   - Testing optimal dopant types and concentrations.
   - Simulating phenomena at a larger scale, such as defect formation and favourable adsorption sites.
     
**3. Methodology**
   - Allegro-FM integration with [Atomic Simulation Environment](https://nequip.readthedocs.io/en/latest/integrations/ase.html): Use ASE package to build the single layer bulk and defective graphene.
   - Data Preparation for Fine-Tuning: Gather dataset of graphene or doped graphene structures and their corresponding energies and forces to use for fine-tuning.
   - Model Fine-tuning: Load pre-trained Allegro-FM with 89 elements, then fine-tune the model using the prepared graphene dataset to optimize for this particular material.
   - MD runs: Run [large-scale molecular dynamics simulations](https://www.lammps.org/#gsc.tab=0). Also simulate the effects of doping, such as changes in lattice structure or defect formation.
   - Result Analysis: Use ASE tools for preliminary analysis, then use tools like [phonopy](https://phonopy.github.io/phonopy/phonopy-module.html) or [mdapy](https://mdapy.readthedocs.io/en/latest/) for validation and characterization purpose. 

**4. Expected results**
   - Dopants concentration/types effects on graphene's structure perperties
   - Different defects effects on graphene chemical and physical  properties
