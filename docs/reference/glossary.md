# Glossary of Terms

## Overview

This glossary provides definitions for key terms used throughout the computational drug discovery roadmap documentation. Terms are organized alphabetically within categories for easy reference. External links are provided for deeper learning.

---

## A

**ADMET**
- **Definition**: Absorption, Distribution, Metabolism, Excretion, and Toxicity - key pharmacological properties that determine a drug's success
- **Context**: Critical parameters evaluated during drug development to predict clinical outcomes
- **Resources**: [FDA ADMET Guidelines](https://www.fda.gov/drugs/drug-development-and-drug-approval-process), [SwissADME Tool](http://www.swissadme.ch/)

**Active Learning**
- **Definition**: Machine learning approach where the algorithm selectively queries the most informative examples for training
- **Context**: Used in drug discovery to optimize experimental design and reduce the number of compounds that need to be synthesized and tested
- **Resources**: [Active Learning Review](https://doi.org/10.1016/j.drudis.2019.06.010), [modAL Python Library](https://modal-python.readthedocs.io/)

**Ansatz**
- **Definition**: In quantum computing, an educated guess for the form of a quantum circuit or wavefunction
- **Context**: Used in variational quantum algorithms like VQE to propose trial solutions for molecular systems
- **Resources**: [Qiskit Tutorials](https://qiskit.org/textbook/), [VQE Documentation](https://qiskit.org/documentation/tutorials/algorithms/04_vqe_advanced.html)

---

## B

**Binding Affinity**
- **Definition**: The strength of interaction between a drug molecule and its target protein, typically measured as IC50, Kd, or Ki
- **Context**: Key metric for evaluating drug potency and designing better therapeutics
- **Resources**: [BindingDB](https://www.bindingdb.org/), [PDBbind Database](http://www.pdbbind.org.cn/)

**Bioavailability**
- **Definition**: The fraction of an administered drug that reaches systemic circulation in an unchanged form
- **Context**: Critical ADMET property that determines dosing requirements
- **Resources**: [FDA Bioavailability Guidelines](https://www.fda.gov/regulatory-information/search-fda-guidance-documents), [Biopharmaceutics Modeling](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6161666/)

**Bioisosterism**
- **Definition**: The replacement of one functional group with another that has similar biological activity
- **Context**: Drug design strategy used to optimize pharmacological properties while maintaining efficacy
- **Resources**: [Bioisosterism in Drug Design](https://doi.org/10.1021/jm501297j), [SwissBioisostere Database](http://www.swissbioisostere.ch/)

---

## C

**ChEMBL**
- **Definition**: Large-scale bioactivity database containing drug-like bioactive compounds and their targets
- **Context**: Primary data source for training machine learning models in drug discovery
- **Resources**: [ChEMBL Database](https://www.ebi.ac.uk/chembl/), [ChEMBL Web Services](https://chembl.gitbook.io/chembl-interface-documentation/), [chembl_webresource_client](https://github.com/chembl/chembl_webresource_client)

**Cheminformatics**
- **Definition**: The application of informatics techniques to solve chemical problems, including molecular representation and property prediction
- **Context**: Fundamental field combining chemistry with computer science for drug discovery
- **Resources**: [RDKit Documentation](https://www.rdkit.org/docs/), [Open Source Cheminformatics](https://github.com/PatWalters/practical_cheminformatics_tutorials), [Journal of Cheminformatics](https://jcheminf.biomedcentral.com/)

**Conformation**
- **Definition**: The 3D spatial arrangement of atoms in a molecule that can be interconverted by rotation around single bonds
- **Context**: Important for drug-target interactions as different conformations may have different binding properties
- **Resources**: [OpenEye OMEGA](https://www.eyesopen.com/omega), [RDKit Conformer Generation](https://www.rdkit.org/docs/GettingStartedInPython.html#working-with-3d-molecules)

**Cross-Validation**
- **Definition**: Statistical method for assessing how well a machine learning model generalizes to independent datasets
- **Context**: Essential for evaluating QSAR models and preventing overfitting in drug discovery applications
- **Resources**: [scikit-learn Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), [Cross-validation in QSAR](https://doi.org/10.1021/ci025626i)

---

## D

**DFT (Density Functional Theory)**
- **Definition**: Quantum mechanical method for calculating electronic structure of many-body systems
- **Context**: Widely used computational chemistry method for predicting molecular properties
- **Resources**: [Psi4 Documentation](https://psicode.org/), [DFT Tutorial](https://www.quantum-espresso.org/Doc/pw_user_guide/), [CP2K](https://www.cp2k.org/)

**Drug-Target Interaction (DTI)**
- **Definition**: The binding interaction between a drug molecule and its biological target
- **Context**: Central to drug discovery - understanding and predicting these interactions is key to drug design
- **Resources**: [DrugBank](https://go.drugbank.com/), [STITCH Database](http://stitch.embl.de/), [DTI Prediction Methods](https://doi.org/10.1093/bib/bbz082)

**Druglikeness**
- **Definition**: A qualitative concept used to evaluate how "drug-like" a compound is based on various molecular properties
- **Context**: Used to filter compound libraries and guide drug design decisions
- **Resources**: [Lipinski's Rule of Five](https://doi.org/10.1016/S0169-409X(96)00423-1), [QED Drug-likeness](https://doi.org/10.1038/nchem.1243)

---

## E

**Ensemble Methods**
- **Definition**: Machine learning techniques that combine multiple models to improve prediction accuracy
- **Context**: Often used in QSAR modeling to reduce overfitting and improve robustness
- **Resources**: [scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html), [XGBoost](https://xgboost.readthedocs.io/), [Random Forest in QSAR](https://doi.org/10.1021/ci034160g)

**Excited States**
- **Definition**: Higher energy electronic states of a molecule above the ground state
- **Context**: Important for understanding photochemistry and designing light-activated drugs
- **Resources**: [Time-Dependent DFT](https://doi.org/10.1021/cr9904009), [Q-Chem Excited States](https://www.q-chem.com/)

---

## F

**FBDD (Fragment-Based Drug Discovery)**
- **Definition**: Drug design approach that starts with small chemical fragments and grows them into drug-like molecules
- **Context**: Alternative to high-throughput screening that can explore chemical space more efficiently
- **Resources**: [Fragment-Based Drug Discovery](https://doi.org/10.1038/nrd3926), [FBDD Database](https://www.ebi.ac.uk/chembl/beta/fbdd/)

**Fingerprints (Molecular)**
- **Definition**: Bit vector representations of molecular structure used for similarity searching and machine learning
- **Context**: Common molecular representation in cheminformatics applications
- **Resources**: [RDKit Fingerprints](https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity), [Morgan Fingerprints](https://doi.org/10.1021/ci100050t)

**Force Field**
- **Definition**: Mathematical functions and parameters used to calculate the potential energy of a molecular system
- **Context**: Used in molecular dynamics simulations and conformational analysis
- **Resources**: [AMBER Force Field](https://ambermd.org/), [CHARMM Force Field](https://www.charmm.org/), [OpenMM](http://openmm.org/)

---

## G

**GNN (Graph Neural Network)**
- **Definition**: Deep learning architecture designed to work with graph-structured data like molecules
- **Context**: State-of-the-art method for molecular property prediction and drug design

**GPCR (G-Protein Coupled Receptor)**
- **Definition**: Large family of membrane proteins that are important drug targets
- **Context**: Represent about 30% of all drug targets in current therapeutics

---

## H

**Hit**
- **Definition**: A compound that shows desired biological activity in an initial screening assay
- **Context**: Starting point for drug development that requires further optimization

**HTS (High-Throughput Screening)**
- **Definition**: Automated method for rapidly testing large numbers of compounds for biological activity
- **Context**: Traditional approach for identifying hits in drug discovery

**Hybrid Quantum-Classical**
- **Definition**: Computational approaches that combine quantum and classical computing methods
- **Context**: Current practical approach for quantum computing applications in chemistry

---

## I

**IC50**
- **Definition**: Half-maximal inhibitory concentration - the concentration of inhibitor required to reduce activity by 50%
- **Context**: Standard measure of drug potency

**In Silico**
- **Definition**: Computer simulation or modeling (literally "in silicon")
- **Context**: Computational drug discovery methods as opposed to wet lab experiments

---

## L

**Lead Compound**
- **Definition**: A chemical compound that shows promising biological activity and serves as starting point for drug development
- **Context**: Requires optimization to become a clinical candidate

**Lipinski's Rule of Five**
- **Definition**: Set of rules describing molecular properties associated with oral bioavailability
- **Context**: Guidelines for drug-like properties (MW<500, LogP<5, HBD≤5, HBA≤10)

---

## M

**Molecular Dynamics (MD)**
- **Definition**: Computer simulation method for analyzing physical movements of atoms and molecules
- **Context**: Used to study protein-drug interactions and conformational flexibility

**Molecular Docking**
- **Definition**: Computational technique for predicting the binding mode of a drug molecule to its target
- **Context**: Fundamental method in structure-based drug design

**Multi-Task Learning**
- **Definition**: Machine learning approach where a model learns to perform multiple related tasks simultaneously
- **Context**: Used in drug discovery to predict multiple molecular properties at once

---

## N

**NISQ (Noisy Intermediate-Scale Quantum)**
- **Definition**: Current era of quantum computing with 10-1000 qubits but significant noise
- **Context**: Describes current quantum computers and algorithms designed for them

---

## P

**Pharmacophore**
- **Definition**: The 3D arrangement of chemical features necessary for biological activity
- **Context**: Used in drug design to identify common binding patterns

**PK/PD (Pharmacokinetics/Pharmacodynamics)**
- **Definition**: PK studies drug absorption, distribution, metabolism, excretion; PD studies drug effects on the body
- **Context**: Critical for understanding drug behavior in clinical settings

**PROTAC (PROteolysis TArgeting Chimera)**
- **Definition**: Engineered molecules that recruit target proteins for degradation
- **Context**: Emerging drug modality that represents novel mechanism beyond traditional inhibition

---

## Q

**QAOA (Quantum Approximate Optimization Algorithm)**
- **Definition**: Quantum algorithm for finding approximate solutions to combinatorial optimization problems
- **Context**: Applied to molecular optimization and drug design problems

**QSAR (Quantitative Structure-Activity Relationship)**
- **Definition**: Mathematical relationships between chemical structure and biological activity
- **Context**: Fundamental approach for predicting drug properties from molecular structure

**Qubit**
- **Definition**: Basic unit of quantum information, analogous to classical bits but can exist in superposition
- **Context**: Building blocks of quantum computers used for chemistry calculations

---

## R

**RDKit**
- **Definition**: Open-source cheminformatics toolkit for molecular manipulation and property calculation
- **Context**: Most widely used Python library for computational chemistry applications

---

## S

**SAR (Structure-Activity Relationship)**
- **Definition**: The relationship between chemical structure and biological activity
- **Context**: Guides medicinal chemists in optimizing drug compounds

**SBDD (Structure-Based Drug Design)**
- **Definition**: Drug design approach that uses 3D structure of target proteins
- **Context**: Rational approach to drug design when target structure is available

**SMILES (Simplified Molecular Input Line Entry System)**
- **Definition**: Text representation of molecular structure using ASCII characters
- **Context**: Standard format for representing molecules in databases and machine learning

---

## T

**Target**
- **Definition**: Biological molecule (usually protein) that a drug is designed to interact with
- **Context**: Starting point for drug discovery campaigns

**Transfer Learning**
- **Definition**: Machine learning technique where knowledge from one task is applied to related tasks
- **Context**: Used when training data is limited for specific drug targets

---

## V

**Virtual Screening**
- **Definition**: Computational technique for searching large libraries of compounds for potential drug candidates
- **Context**: Alternative to experimental high-throughput screening

**VQE (Variational Quantum Eigensolver)**
- **Definition**: Quantum algorithm for finding ground state energies of molecular systems
- **Context**: Most promising near-term quantum algorithm for chemistry applications

---

## Usage Guidelines

### For Beginners
- Start with basic terms in each category
- Use the context information to understand practical applications
- Cross-reference with main documentation for detailed explanations

### For Advanced Users
- Use as quick reference during reading
- Contribute additional terms and definitions
- Suggest improvements to existing definitions

### For Instructors
- Use as teaching reference
- Assign students to create definitions for new terms
- Incorporate into assessment materials

---

## Contributing to the Glossary

### How to Add Terms
1. Follow the existing format (Term, Definition, Context)
2. Place in appropriate alphabetical position
3. Include practical context relevant to drug discovery
4. Keep definitions concise but comprehensive

### Quality Guidelines
- **Accuracy**: Ensure definitions are scientifically correct
- **Clarity**: Write for diverse audiences
- **Completeness**: Include sufficient context
- **Consistency**: Use consistent terminology throughout

---

## Related Resources

- [Quick Reference Card](./quick_reference_card.md) - Essential commands and workflows
- [API Reference](./api_reference.md) - Technical function definitions
- [Main Roadmap](../roadmaps/unified_roadmap.md) - Learning context for terms
- [Resource Collection](../resources/comprehensive_resource_collection.md) - Additional learning materials

---

*This glossary is continuously updated. If you encounter unfamiliar terms not listed here, please suggest additions through the documentation feedback process.*
