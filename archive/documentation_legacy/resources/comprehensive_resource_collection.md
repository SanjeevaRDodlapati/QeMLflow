# Comprehensive Resource Collection for Computational Drug Discovery

## Overview
This document provides a curated collection of resources, tools, datasets, and educational materials for computational drug discovery research. All resources are organized by category with direct links, descriptions, and usage recommendations.

## Table of Contents
1. [Software Tools & Platforms](#software-tools--platforms)
2. [Databases & Datasets](#databases--datasets)
3. [Educational Resources](#educational-resources)
4. [Hardware & Computing Resources](#hardware--computing-resources)
5. [Professional Development](#professional-development)
6. [Research Communities](#research-communities)
7. [Funding Opportunities](#funding-opportunities)
8. [Literature & References](#literature--references)

---

## Software Tools & Platforms

### Cheminformatics & Molecular Modeling

#### Core Cheminformatics Libraries
| Tool | Description | Language | License | Installation |
|------|-------------|----------|---------|-------------|
| [RDKit](https://www.rdkit.org/) | Open-source cheminformatics toolkit | Python/C++ | BSD | `conda install -c rdkit rdkit` |
| [Open Babel](http://openbabel.org/) | Chemical file format converter | C++/Python | GPL | `conda install -c openbabel openbabel` |
| [ChemPy](https://github.com/bjodah/chempy) | Python package for chemistry | Python | BSD | `pip install chempy` |
| [Mordred](https://github.com/mordred-descriptor/mordred) | Molecular descriptor calculator | Python | BSD | `pip install mordred` |
| [DeepChem](https://deepchem.io/) | Deep learning for drug discovery | Python | MIT | `pip install deepchem` |

#### Molecular Visualization
| Tool | Description | Platform | License | URL |
|------|-------------|----------|---------|-----|
| [PyMOL](https://pymol.org/) | Molecular visualization system | Cross-platform | Commercial/Open | https://pymol.org/ |
| [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) | Next-generation molecular visualization | Cross-platform | Free | https://www.cgl.ucsf.edu/chimerax/ |
| [VMD](https://www.ks.uiuc.edu/Research/vmd/) | Visual Molecular Dynamics | Cross-platform | Free | https://www.ks.uiuc.edu/Research/vmd/ |
| [NGLView](https://github.com/nglviewer/nglview) | Jupyter notebook molecular viewer | Python/Web | MIT | `pip install nglview` |
| [py3Dmol](https://github.com/3dmol/3Dmol.js) | 3D molecular visualization in Python | Python/Web | MIT | `pip install py3dmol` |

#### Molecular Dynamics Simulation
| Tool | Description | Specialty | License | Documentation |
|------|-------------|-----------|---------|---------------|
| [GROMACS](http://www.gromacs.org/) | High-performance MD package | General MD | GPL | http://manual.gromacs.org/ |
| [AMBER](https://ambermd.org/) | Biomolecular simulation suite | Biomolecules | Commercial | https://ambermd.org/doc12/ |
| [NAMD](https://www.ks.uiuc.edu/Research/namd/) | Scalable MD for large systems | Large systems | Free | https://www.ks.uiuc.edu/Research/namd/ |
| [OpenMM](http://openmm.org/) | High-performance MD toolkit | GPU acceleration | MIT | http://docs.openmm.org/ |
| [CHARMM](https://www.charmm.org/) | Chemistry at HARvard MM | Force fields | Commercial | https://www.charmm.org/ |

#### Quantum Chemistry
| Software | Description | Methods | License | Installation |
|----------|-------------|---------|---------|-------------|
| [Gaussian](https://gaussian.com/) | Electronic structure package | DFT, ab initio | Commercial | Licensed |
| [ORCA](https://orcaforum.kofo.mpg.de/) | Ab initio quantum chemistry | Multi-reference | Free (academic) | Download |
| [PySCF](https://pyscf.org/) | Python-based quantum chemistry | All methods | Apache | `pip install pyscf` |
| [Psi4](http://www.psicode.org/) | Open-source quantum chemistry | DFT, CC, CI | GPL | `conda install psi4` |
| [Q-Chem](https://www.q-chem.com/) | Quantum chemistry package | Time-dependent DFT | Commercial | Licensed |

### Machine Learning & AI

#### Deep Learning Frameworks
| Framework | Description | Specialty | Python API | Documentation |
|-----------|-------------|-----------|------------|---------------|
| [PyTorch](https://pytorch.org/) | Dynamic neural networks | Research flexibility | Native | https://pytorch.org/docs/ |
| [TensorFlow](https://tensorflow.org/) | End-to-end ML platform | Production deployment | Native | https://tensorflow.org/api_docs |
| [JAX](https://github.com/google/jax) | NumPy-compatible ML | High-performance | Native | https://jax.readthedocs.io/ |
| [Flax](https://github.com/google/flax) | JAX-based neural networks | Functional programming | JAX | https://flax.readthedocs.io/ |

#### Specialized ML for Chemistry
| Tool | Description | Focus Area | Framework | Repository |
|------|-------------|------------|-----------|------------|
| [DGL-LifeSci](https://github.com/awslabs/dgl-lifesci) | Graph neural networks for life sciences | Graph ML | PyTorch/DGL | https://github.com/awslabs/dgl-lifesci |
| [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) | Graph neural networks | Graph deep learning | PyTorch | https://github.com/pyg-team/pytorch_geometric |
| [Chemprop](https://github.com/chemprop/chemprop) | Message passing neural networks | Molecular property prediction | PyTorch | https://github.com/chemprop/chemprop |
| [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack) | Neural networks for atomistic systems | Quantum chemistry ML | PyTorch | https://github.com/atomistic-machine-learning/schnetpack |

### Quantum Computing

#### Quantum Development Frameworks
| Platform | Company | Language | Simulators | Hardware Access |
|----------|---------|----------|------------|-----------------|
| [Qiskit](https://qiskit.org/) | IBM | Python | Aer | IBM Quantum |
| [Cirq](https://quantumai.google/cirq) | Google | Python | Multiple | Google Quantum AI |
| [PennyLane](https://pennylane.ai/) | Xanadu | Python | Multiple | Various providers |
| [Forest SDK](https://github.com/rigetti/pyquil) | Rigetti | Python | QVM | Rigetti Quantum Cloud |
| [Q#](https://azure.microsoft.com/en-us/products/quantum) | Microsoft | Q# | Local simulator | Azure Quantum |

#### Quantum Chemistry Software
| Tool | Description | Methods | Integration | Documentation |
|------|-------------|---------|-------------|---------------|
| [OpenFermion](https://github.com/quantumlib/OpenFermion) | Quantum simulation of fermionic systems | VQE, QAOA | Cirq, Qiskit | https://github.com/quantumlib/OpenFermion |
| [Qiskit Nature](https://qiskit.org/ecosystem/nature/) | Quantum algorithms for natural sciences | Ground state, excited states | Qiskit | https://qiskit.org/documentation/nature/ |
| [PennyLane-QChem](https://github.com/PennyLaneAI/pennylane-qchem) | Quantum chemistry with PennyLane | Molecular Hamiltonians | PennyLane | https://pennylane-qchem.readthedocs.io/ |

### Drug Discovery Platforms

#### Commercial Platforms
| Platform | Company | Features | Pricing | Trial Available |
|----------|---------|----------|---------|----------------|
| [Schrödinger Suite](https://www.schrodinger.com/) | Schrödinger | Complete drug discovery | Commercial license | Academic discounts |
| [MOE](https://www.chemcomp.com/Products.htm) | Chemical Computing Group | Molecular modeling | Commercial license | Demo available |
| [Pipeline Pilot](https://www.3ds.com/products-services/biovia/products/data-science/pipeline-pilot/) | BIOVIA | Data pipelines | Commercial license | Trial available |
| [ChemAxon](https://chemaxon.com/) | ChemAxon | Cheminformatics tools | Various licenses | Free academic |

#### Open-Source Platforms
| Platform | Description | Features | Language | Repository |
|----------|-------------|----------|----------|------------|
| [KNIME](https://www.knime.com/) | Data analytics platform | Visual workflows | Java | Open source + commercial |
| [Orange](https://orangedatamining.com/) | Data mining toolkit | Visual programming | Python | https://github.com/biolab/orange3 |
| [OSIRIS](https://openmolecules.org/datawarrior/) | Cheminformatics platform | Property prediction | Java | Free download |

---

## Databases & Datasets

### Chemical Databases

#### Public Chemical Databases
| Database | Description | Size | Access | API Available |
|----------|-------------|------|--------|---------------|
| [PubChem](https://pubchem.ncbi.nlm.nih.gov/) | Chemical information database | 100M+ compounds | Free | Yes |
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | Bioactive molecules database | 2M+ compounds | Free | Yes |
| [ZINC](https://zinc.docking.org/) | Purchasable compounds | 230M+ compounds | Free | Yes |
| [ChemSpider](http://www.chemspider.com/) | Chemical structure database | 100M+ structures | Free | Yes |
| [DrugBank](https://go.drugbank.com/) | Comprehensive drug database | 13K+ drugs | Free/Commercial | Yes |

#### Commercial Chemical Databases
| Database | Company | Specialty | Access Model | Academic Pricing |
|----------|---------|-----------|--------------|------------------|
| [Reaxys](https://www.reaxys.com/) | Elsevier | Chemical reactions | Subscription | Academic licenses |
| [SciFinder](https://scifinder.cas.org/) | CAS | Literature + chemicals | Subscription | Academic rates |
| [eMolecules](https://www.emolecules.com/) | eMolecules | Purchasable compounds | Free/Subscription | Academic discounts |

### Biological Databases

#### Protein Databases
| Database | Description | Content | Format | Download |
|----------|-------------|---------|--------|----------|
| [Protein Data Bank (PDB)](https://www.rcsb.org/) | 3D protein structures | 190K+ structures | PDB, mmCIF | https://www.rcsb.org/ |
| [UniProt](https://www.uniprot.org/) | Protein sequence and function | 200M+ proteins | FASTA, XML | https://www.uniprot.org/ |
| [Pfam](https://pfam.xfam.org/) | Protein families | 19K+ families | Multiple | https://pfam.xfam.org/ |
| [SCOP](https://scop.mrc-lmb.cam.ac.uk/) | Structural classification | Hierarchical | Text, XML | https://scop.mrc-lmb.cam.ac.uk/ |

#### Bioactivity Databases
| Database | Description | Data Type | Size | Access |
|----------|-------------|-----------|------|--------|
| [BindingDB](https://www.bindingdb.org/) | Binding affinities | IC50, Ki, Kd | 2M+ measurements | Free |
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | Bioactivity data | Multiple assays | 19M+ activities | Free |
| [PubChem BioAssay](https://pubchem.ncbi.nlm.nih.gov/bioassay/) | Biological screenings | HTS data | 1M+ assays | Free |
| [DTC](https://drugtargetcommons.fimm.fi/) | Drug-target interactions | Curated DTIs | 900K+ interactions | Free |

### Specialized Datasets

#### ADMET Datasets
| Dataset | Property | Size | Source | Paper |
|---------|----------|------|--------|--------|
| [ToxCast](https://www.epa.gov/chemical-research/toxicity-forecaster-toxcasttm-data) | Toxicity | 9K compounds | EPA | Nature Biotechnology 2017 |
| [Tox21](https://tripod.nih.gov/tox21/) | Toxicity pathways | 10K compounds | NIH | Chemical Research in Toxicology 2014 |
| [DILI](https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb) | Drug-induced liver injury | 1K+ drugs | FDA | Chemical Research in Toxicology 2016 |
| [hERG](https://pubchem.ncbi.nlm.nih.gov/bioassay/376) | Cardiotoxicity | 5K+ compounds | PubChem | Multiple |

#### Machine Learning Benchmark Datasets
| Dataset | Description | Task Type | Size | Reference |
|---------|-------------|-----------|------|-----------|
| [MoleculeNet](http://moleculenet.ai/) | ML benchmarking suite | Multiple | Various | Chemical Science 2018 |
| [TDC](https://tdcommons.ai/) | Therapeutics Data Commons | Drug discovery | Multiple datasets | Nature Chemical Biology 2021 |
| [QM7/QM9](http://quantum-machine.org/datasets/) | Quantum chemistry | Property prediction | 7K/134K molecules | Scientific Data 2014/2017 |
| [PCBA](https://pubchem.ncbi.nlm.nih.gov/pc_fetch/pc_fetch.cgi) | PubChem BioAssay | Classification | 400K+ compounds | Various |

---

## Educational Resources

### Online Courses

#### Machine Learning for Chemistry
| Course | Provider | Duration | Level | Certificate |
|--------|----------|----------|-------|-------------|
| [Machine Learning for Drug Discovery](https://www.edx.org/course/machine-learning-for-drug-discovery) | MIT (edX) | 12 weeks | Advanced | Yes |
| [Deep Learning for Molecules and Materials](https://dmol.pub/) | Carnegie Mellon | Self-paced | Intermediate | Free |
| [Computational Drug Discovery](https://www.coursera.org/learn/computational-drug-discovery) | University of California San Diego | 6 weeks | Intermediate | Yes |
| [Cheminformatics](https://www.futurelearn.com/courses/cheminformatics) | University of Leeds | 4 weeks | Beginner | Yes |

#### Quantum Computing
| Course | Provider | Focus | Duration | Prerequisites |
|--------|----------|-------|----------|---------------|
| [Qiskit Textbook](https://qiskit.org/textbook/) | IBM | Quantum algorithms | Self-paced | Linear algebra |
| [Microsoft Quantum Katas](https://github.com/Microsoft/QuantumKatas) | Microsoft | Q# programming | Self-paced | Programming |
| [Quantum Computing for Computer Scientists](https://www.cambridge.org/core/books/quantum-computing/8AEA723BEE5CC9F5C03FDD4BA850C711) | Cambridge | Theory | Book | Mathematics |
| [Quantum Machine Learning](https://pennylane.ai/qml/) | Xanadu | QML algorithms | Self-paced | ML + QC basics |

#### Programming and Data Science
| Course | Provider | Technology | Level | Duration |
|--------|----------|------------|-------|----------|
| [Python for Data Science](https://www.coursera.org/learn/python-data-science) | University of Michigan | Python/Pandas | Beginner | 5 weeks |
| [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) | deeplearning.ai | TensorFlow/PyTorch | Intermediate | 16 weeks |
| [Fast.ai Practical Deep Learning](https://www.fast.ai/) | fast.ai | PyTorch | Beginner-Advanced | Self-paced |
| [CS224W: Graph Neural Networks](http://web.stanford.edu/class/cs224w/) | Stanford | Graph ML | Advanced | 10 weeks |

### Books and Textbooks

#### Drug Discovery and Cheminformatics
| Title | Author(s) | Publisher | Year | Focus |
|-------|-----------|-----------|------|-------|
| "Deep Learning for the Life Sciences" | Ramsundar et al. | O'Reilly | 2019 | ML applications |
| "Computational Drug Discovery and Design" | Baron (Ed.) | Springer | 2012 | Methods overview |
| "Introduction to Cheminformatics" | Leach & Gillet | Springer | 2007 | Fundamentals |
| "Molecular Descriptors for Chemoinformatics" | Todeschini & Consonni | Wiley-VCH | 2009 | Descriptors |

#### Machine Learning and AI
| Title | Author(s) | Publisher | Year | Relevance |
|-------|-----------|-----------|------|-----------|
| "Pattern Recognition and Machine Learning" | Bishop | Springer | 2006 | ML theory |
| "Deep Learning" | Goodfellow et al. | MIT Press | 2016 | Deep learning |
| "Hands-On Machine Learning" | Géron | O'Reilly | 2019 | Practical ML |
| "Graph Representation Learning" | Hamilton | Morgan & Claypool | 2020 | Graph ML |

#### Quantum Computing
| Title | Author(s) | Publisher | Year | Level |
|-------|-----------|-----------|------|-------|
| "Quantum Computing: An Applied Approach" | Hidary | Springer | 2021 | Practical |
| "Programming Quantum Computers" | Johnston et al. | O'Reilly | 2019 | Programming |
| "Quantum Computation and Quantum Information" | Nielsen & Chuang | Cambridge | 2010 | Theoretical |

### Tutorials and Workshops

#### Hands-on Tutorials
| Tutorial | Provider | Topic | Format | Link |
|----------|----------|-------|--------|------|
| [RDKit Cookbook](https://rdkit.readthedocs.io/en/latest/Cookbook.html) | RDKit | Cheminformatics | Code examples | https://rdkit.readthedocs.io/ |
| [DeepChem Tutorials](https://github.com/deepchem/deepchem/tree/master/examples/tutorials) | DeepChem | ML for chemistry | Jupyter notebooks | https://github.com/deepchem/deepchem |
| [Qiskit Tutorials](https://qiskit.org/documentation/tutorials.html) | IBM | Quantum computing | Jupyter notebooks | https://qiskit.org/documentation/ |
| [PyTorch Geometric Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html) | PyG Team | Graph neural networks | Code examples | https://pytorch-geometric.readthedocs.io/ |

#### Video Lecture Series
| Series | Instructor | Institution | Topic | Platform |
|--------|------------|-------------|--------|----------|
| [Machine Learning for Drug Discovery](https://www.youtube.com/playlist?list=PLypiXJdtIca5ElZMWHl4HMeyle2AzUgVB) | Rafael Gómez-Bombarelli | MIT | ML applications | YouTube |
| [Quantum Machine Learning](https://www.youtube.com/playlist?list=PLmRxgFnCIhaMgvot-Xuym_hn69lmzIokg) | Maria Schuld | Xanadu | QML | YouTube |
| [Graph Neural Networks](https://www.youtube.com/playlist?list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z) | Jure Leskovec | Stanford | Graph ML | YouTube |

---

## Hardware & Computing Resources

### High-Performance Computing

#### Academic HPC Centers
| Center | Institution | Access | Resources | Application |
|--------|-------------|--------|-----------|-------------|
| [XSEDE](https://www.xsede.org/) | NSF | Allocation | Supercomputers | https://www.xsede.org/ |
| [NERSC](https://www.nersc.gov/) | DOE | ERCAP | Cori, Perlmutter | https://www.nersc.gov/ |
| [ACCESS](https://access-ci.org/) | NSF | Allocation | Multiple centers | https://access-ci.org/ |
| [Compute Canada](https://www.computecanada.ca/) | Canada | RAC | Cedar, Graham, Niagara | https://www.computecanada.ca/ |

#### Cloud Computing Platforms
| Provider | Quantum Access | GPU Options | Chemistry Software | Pricing Model |
|----------|----------------|-------------|-------------------|---------------|
| [AWS](https://aws.amazon.com/) | Braket | EC2 P3/P4 | Containers | Pay-per-use |
| [Google Cloud](https://cloud.google.com/) | Quantum AI | Compute Engine | Docker | Pay-per-use |
| [Microsoft Azure](https://azure.microsoft.com/) | Quantum | ND-series | Containers | Pay-per-use |
| [IBM Cloud](https://www.ibm.com/cloud) | Quantum Network | V100/A100 | Watson | Pay-per-use |

### Quantum Computing Access

#### Quantum Cloud Services
| Provider | Hardware Type | Qubits | Programming | Access Model |
|----------|---------------|--------|-------------|--------------|
| [IBM Quantum](https://quantum-computing.ibm.com/) | Superconducting | 5-1000+ | Qiskit | Free + premium |
| [Google Quantum AI](https://quantumai.google/) | Superconducting | 70+ | Cirq | Research access |
| [IonQ](https://ionq.com/) | Trapped ion | 32+ | Multiple | Cloud access |
| [Rigetti](https://www.rigetti.com/) | Superconducting | 80+ | PyQuil | Cloud access |
| [Xanadu](https://www.xanadu.ai/) | Photonic | Variable | PennyLane | Cloud access |

#### Quantum Simulators
| Simulator | Type | Performance | Integration | License |
|-----------|------|-------------|-------------|---------|
| [Qiskit Aer](https://qiskit.org/aer/) | Multiple | High | Qiskit | Apache |
| [Cirq Simulator](https://quantumai.google/cirq) | State vector | Medium | Cirq | Apache |
| [PennyLane](https://pennylane.ai/) | Multiple | High | Multiple | Apache |
| [QuTiP](http://qutip.org/) | Open systems | Medium | Python | BSD |

---

## Professional Development

### Conferences and Workshops

#### Major Conferences
| Conference | Focus Area | Frequency | Location | Deadline |
|------------|------------|-----------|----------|----------|
| [AAAI](https://aaai.org/Conferences/AAAI-23/) | Artificial Intelligence | Annual | Rotating | August |
| [ICML](https://icml.cc/) | Machine Learning | Annual | International | February |
| [NeurIPS](https://neurips.cc/) | Neural Information Processing | Annual | International | May |
| [ACS National Meeting](https://www.acs.org/content/acs/en/meetings.html) | Chemistry | Biannual | USA | Various |
| [ISMB](https://www.iscb.org/ismb2023) | Bioinformatics | Annual | International | February |

#### Specialized Workshops
| Workshop | Focus | Frequency | Format | Registration |
|----------|-------|-----------|--------|-------------|
| [ML4Molecules](https://ml4molecules.github.io/) | ML for chemistry | Annual | 1-2 days | Usually with major conference |
| [CADD Gordon Conference](https://www.grc.org/computer-aided-drug-design-conference/) | Drug design | Biennial | 5 days | Invitation/application |
| [QC4D Workshop](https://www.quantum-computing-for-drug-discovery.com/) | Quantum + pharma | Annual | 2 days | Open registration |
| [AI in Drug Discovery](https://www.ai-in-drug-discovery.com/) | AI applications | Annual | 2-3 days | Commercial |

#### Quantum Computing Events
| Event | Organizer | Focus | Format | Audience |
|-------|-----------|-------|--------|---------|
| [QIP](https://qip2023.github.io/) | International | Quantum information | Academic conference | Researchers |
| [Qiskit Global Summer School](https://qiskit.org/events/summer-school/) | IBM | Education | 2-week virtual | Students |
| [Q2B](https://q2b.qcware.com/) | QC Ware | Business applications | Commercial conference | Industry |

### Professional Societies

#### Scientific Societies
| Society | Focus Area | Membership Benefits | Student Rates | Website |
|---------|------------|-------------------|---------------|---------|
| [American Chemical Society (ACS)](https://www.acs.org/) | Chemistry | Publications, networking | Yes | https://www.acs.org/ |
| [Association for Computing Machinery (ACM)](https://www.acm.org/) | Computing | Digital library, conferences | Yes | https://www.acm.org/ |
| [IEEE](https://www.ieee.org/) | Engineering | Standards, publications | Yes | https://www.ieee.org/ |
| [American Physical Society (APS)](https://www.aps.org/) | Physics | Journals, meetings | Yes | https://www.aps.org/ |

#### Specialized Organizations
| Organization | Focus | Benefits | Membership Type | Application |
|--------------|-------|----------|----------------|-------------|
| [International Society for Computational Biology (ISCB)](https://www.iscb.org/) | Computational biology | Conferences, training | Open | Online |
| [Computer-Aided Drug Design (CADD) Group](https://www.cadd-group.org/) | Drug design | Networking, workshops | Invitation | Application |
| [Quantum Economic Development Consortium (QED-C)](https://quantumconsortium.org/) | Quantum industry | Industry connections | Corporate/individual | Application |

### Career Services and Mentorship

#### Mentorship Programs
| Program | Target Audience | Duration | Matching Process | Cost |
|---------|----------------|----------|------------------|------|
| [ACS Career Consultant Program](https://www.acs.org/content/acs/en/careers/career-services/career-consultants.html) | Chemists | Ongoing | Self-selected | Free |
| [Women in AI Mentorship](https://www.womeninai.co/mentorship) | AI researchers | 6 months | Application | Free |
| [Quantum Open Source Foundation Mentorship](https://qosf.org/mentorship/) | Quantum developers | 3 months | Application | Free |

#### Career Development Resources
| Resource | Provider | Type | Content | Access |
|----------|----------|------|---------|--------|
| [ACS Career Services](https://www.acs.org/content/acs/en/careers.html) | ACS | Multiple | Job search, CV help | Members |
| [Nature Careers](https://www.nature.com/naturecareers/) | Nature | Online | Career advice, jobs | Free |
| [Science Careers](https://www.sciencecareers.org/) | AAAS | Online | Career guidance | Free |
| [IEEE Job Site](https://jobs.ieee.org/) | IEEE | Job board | Technical positions | Free |

---

## Research Communities

### Online Communities

#### Forums and Discussion Platforms
| Platform | Focus | Format | Activity Level | Moderation |
|----------|-------|--------|---------------|------------|
| [RDKit Discussions](https://github.com/rdkit/rdkit/discussions) | RDKit usage | GitHub Discussions | High | Community |
| [DeepChem Forums](https://forum.deepchem.io/) | DeepChem/ML | Discourse | Medium | Moderated |
| [Qiskit Slack](https://qiskit.slack.com/) | Quantum computing | Slack | High | IBM team |
| [r/MachineLearning](https://www.reddit.com/r/MachineLearning/) | ML research | Reddit | Very high | Community |
| [Biostars](https://www.biostars.org/) | Bioinformatics | Q&A format | High | Community |

#### Social Media and Networking
| Platform | Type | Benefits | Target Audience | Best Practices |
|----------|------|----------|----------------|----------------|
| [Twitter/X](https://twitter.com/) | Microblogging | Real-time updates | Researchers | Follow conferences, share work |
| [LinkedIn](https://www.linkedin.com/) | Professional network | Career opportunities | Professionals | Complete profile, engage |
| [ResearchGate](https://www.researchgate.net/) | Academic network | Paper sharing | Academics | Upload publications |
| [Google Scholar](https://scholar.google.com/) | Citation tracking | Impact metrics | Researchers | Keep profile updated |

### Research Groups and Labs

#### Leading Academic Groups
| Lab/Group | Institution | PI | Research Focus | Notable Contributions |
|-----------|-------------|----|--------------|--------------------|
| [Aspuru-Guzik Group](https://www.matter.toronto.edu/) | University of Toronto | Alán Aspuru-Guzik | Quantum ML, materials | Quantum computing for chemistry |
| [Coley Group](https://coleygroup.mit.edu/) | MIT | Connor Coley | ML for synthesis | Retrosynthesis prediction |
| [Gómez-Bombarelli Group](https://gomezbombarelli.mit.edu/) | MIT | Rafael Gómez-Bombarelli | Molecular ML | Generative models |
| [Ramsundar Group](https://rbharath.github.io/) | Independent | Bharath Ramsundar | DeepChem | Open-source drug discovery |

#### Industry Research Labs
| Lab | Company | Focus | Notable Projects | Collaboration |
|-----|---------|-------|------------------|-------------|
| [Google DeepMind](https://deepmind.com/) | Alphabet | AI research | AlphaFold, AlphaChemistry | Academic partnerships |
| [IBM Research](https://www.ibm.com/research/) | IBM | Quantum + AI | Quantum computing | Open source contributions |
| [Microsoft Research](https://www.microsoft.com/en-us/research/) | Microsoft | Quantum development | Q# development | Academic collaborations |
| [Mila](https://mila.quebec/en/) | Independent | AI research | Graph neural networks | Open research |

---

## Funding Opportunities

### Graduate and Fellowships

#### National Fellowships (US)
| Fellowship | Agency | Duration | Stipend | Eligibility | Deadline |
|------------|--------|----------|---------|-------------|-----------|
| [NSF Graduate Research Fellowship (GRFP)](https://www.nsfgrfp.org/) | NSF | 3 years | $37K/year | Graduate students | October |
| [NIH F31/F32](https://www.nihlibrary.nih.gov/resources/subject-guides/finding-and-applying-nih-funding/individual-fellowships) | NIH | 2-3 years | $25-50K/year | Students/participants | Multiple |
| [DOE SCGSR](https://science.osti.gov/wdts/scgsr) | DOE | 3-12 months | Travel + stipend | Graduate students | November/May |
| [Hertz Fellowship](https://www.hertzfoundation.org/) | Hertz Foundation | 5 years | Full support | PhD students | October |

#### International Fellowships
| Fellowship | Region/Country | Duration | Coverage | Application | Deadline |
|------------|----------------|----------|----------|-------------|----------|
| [Marie Curie Fellowships](https://marie-sklodowska-curie-actions.ec.europa.eu/) | Europe | 1-3 years | Full support | Competitive | September |
| [JSPS Fellowship](https://www.jsps.go.jp/english/e-fellow/) | Japan | 1-2 years | Living allowance | Application | Multiple |
| [Mitacs Fellowship](https://www.mitacs.ca/) | Canada | 4-24 months | Stipend | Partnership | Rolling |
| [DAAD Fellowship](https://www.daad.org/en/) | Germany | Variable | Support | Application | Multiple |

### Research Grants

#### Early Career Grants
| Grant | Agency | Amount | Duration | Career Stage | Focus |
|-------|--------|--------|----------|-------------|-------|
| [NSF CAREER](https://www.nsf.gov/funding/pgm_summ.jsp?pims_id=503214) | NSF | $400-800K | 5 years | Early faculty | Education + research |
| [NIH R01](https://grants.nih.gov/grants/funding/r01.htm) | NIH | $250K/year | 3-5 years | Independent investigators | Health-related |
| [DOE Early Career](https://science.osti.gov/early-career) | DOE | $150K/year | 5 years | Early faculty | Energy sciences |
| [Sloan Fellowship](https://sloan.org/fellowships/) | Sloan Foundation | $75K | 2 years | Early faculty | Basic research |

#### Industry and Foundation Grants
| Grant | Source | Amount | Focus | Application Process | Duration |
|-------|--------|--------|-------|-------------------|----------|
| [Google Research Awards](https://research.google/outreach/past-programs/research-awards/) | Google | $50-150K | AI/ML research | Invitation | 1 year |
| [Microsoft Research Awards](https://www.microsoft.com/en-us/research/academic-program/) | Microsoft | Variable | Computing research | Application | 1-2 years |
| [Chan Zuckerberg Initiative](https://chanzuckerberg.com/science/) | CZI | Variable | Science technology | RFP | Variable |
| [Gordon and Betty Moore Foundation](https://www.moore.org/) | Moore Foundation | $50K-1.5M | Data-driven discovery | Application | 1-5 years |

### Small Grants and Seed Funding

#### Conference and Travel Support
| Grant | Source | Amount | Purpose | Eligibility | Application |
|-------|--------|--------|---------|-------------|-------------|
| [NSF Student Travel Grants](https://www.nsf.gov/funding/education.jsp?fund_type=3) | NSF | $500-2K | Conference travel | Students | Via conference |
| [ACM Travel Grants](https://www.sigplan.org/PAC/) | ACM | $500-1.5K | Conference attendance | Students | Application |
| [ACS Travel Awards](https://www.acs.org/content/acs/en/funding-and-awards/awards/national/nominations.html) | ACS | $500-2K | Meeting attendance | Members | Application |

#### Equipment and Software
| Grant | Source | Amount | Purpose | Eligibility | Notes |
|-------|--------|--------|---------|-------------|-------|
| [NSF MRI](https://www.nsf.gov/funding/pgm_summ.jsp?pims_id=5260) | NSF | $100K-4M | Major instruments | Institutions | Shared use required |
| [Academic Software Discounts](https://www.schrodinger.com/academic) | Various vendors | 50-90% off | Software licenses | Academic | Varies by vendor |
| [Cloud Computing Credits](https://cloud.google.com/edu/researchers) | Cloud providers | $5K-100K | Computing resources | Researchers | Application required |

---

## Literature & References

### Key Journals

#### Drug Discovery and Cheminformatics
| Journal | Publisher | Impact Factor | Focus | Open Access |
|---------|-----------|---------------|-------|-------------|
| [Journal of Medicinal Chemistry](https://pubs.acs.org/journal/jmcmar) | ACS | 7.3 | Medicinal chemistry | Subscription |
| [Journal of Chemical Information and Modeling](https://pubs.acs.org/journal/jcisd8) | ACS | 4.6 | Computational chemistry | Subscription |
| [Drug Discovery Today](https://www.sciencedirect.com/journal/drug-discovery-today) | Elsevier | 7.4 | Drug discovery | Subscription |
| [Journal of Computer-Aided Molecular Design](https://link.springer.com/journal/10822) | Springer | 3.0 | CADD | Subscription |
| [Molecular Informatics](https://onlinelibrary.wiley.com/journal/18681751) | Wiley | 2.8 | Cheminformatics | Subscription |

#### Machine Learning and AI
| Journal | Publisher | Impact Factor | Focus | Access Model |
|---------|-----------|---------------|-------|-------------|
| [Nature Machine Intelligence](https://www.nature.com/natmachintell/) | Nature | 25.9 | AI research | Subscription |
| [Journal of Machine Learning Research](https://www.jmlr.org/) | MIT Press | 4.3 | ML theory | Open access |
| [Machine Learning](https://link.springer.com/journal/10994) | Springer | 3.3 | ML methods | Subscription |
| [IEEE Transactions on Pattern Analysis and Machine Intelligence](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | IEEE | 17.9 | Pattern recognition | Subscription |

#### Quantum Computing
| Journal | Publisher | Impact Factor | Focus | Notes |
|---------|-----------|---------------|-------|-------|
| [Nature Quantum Information](https://www.nature.com/npjqi/) | Nature | 10.8 | Quantum information | Open access |
| [Quantum](https://quantum-journal.org/) | Verein zur Förderung des Open Access Publizierens | 5.1 | Quantum science | Open access |
| [Physical Review Quantum](https://journals.aps.org/prquantum/) | APS | 5.0 | Quantum physics | Open access |
| [npj Quantum Information](https://www.nature.com/npjqi/) | Nature | 6.2 | Quantum technologies | Open access |

### Essential Review Papers

#### Machine Learning for Drug Discovery
| Title | Authors | Journal | Year | Citations | Focus |
|-------|---------|---------|------|-----------|-------|
| "Deep learning for drug discovery" | Chen et al. | Chemical Science | 2018 | 1200+ | ML overview |
| "Molecular machine learning" | Dral | Journal of Physical Chemistry Letters | 2020 | 800+ | Quantum chemistry ML |
| "Graph neural networks for drug discovery" | Wu et al. | Drug Discovery Today | 2021 | 500+ | Graph ML |
| "Transformers in chemistry" | Ross et al. | Nature Machine Intelligence | 2022 | 200+ | Transformer models |

#### Quantum Computing for Chemistry
| Title | Authors | Journal | Year | Impact | Topic |
|-------|---------|---------|------|--------|-------|
| "Quantum computational chemistry" | Cao et al. | Chemical Reviews | 2019 | 1000+ | Comprehensive review |
| "Quantum algorithms for chemistry" | McArdle et al. | Reviews of Modern Physics | 2020 | 800+ | Algorithm survey |
| "Near-term quantum algorithms" | Bharti et al. | Reviews of Modern Physics | 2022 | 400+ | NISQ applications |

### Benchmark Papers and Datasets

#### ML Benchmarking Studies
| Paper | Dataset/Benchmark | Task | Metrics | Significance |
|-------|-------------------|------|---------|-------------|
| "MoleculeNet" (Wu et al., 2018) | MoleculeNet | Multiple | Various | Standard benchmark suite |
| "Therapeutics Data Commons" (Huang et al., 2021) | TDC | Drug discovery | AUROC, etc. | Comprehensive platform |
| "A Deep Learning Approach to Antibiotic Discovery" (Stokes et al., 2020) | Custom | Antibiotic discovery | Experimental validation | Real-world impact |

#### Quantum Computing Benchmarks
| Paper | System/Algorithm | Problem | Advantage Claimed | Verification |
|-------|-------------------|---------|------------------|-------------|
| "Quantum supremacy using a programmable superconducting processor" (Arute et al., 2019) | Google Sycamore | Random sampling | 200 seconds vs 10,000 years | Debated |
| "Quantum computational advantage using photons" (Zhong et al., 2020) | Photonic system | Gaussian boson sampling | Minutes vs billions of years | Verified |

### Software Documentation and Tutorials

#### Comprehensive Guides
| Resource | Type | Coverage | Maintenance | Community |
|----------|------|----------|-------------|-----------|
| [RDKit Documentation](https://rdkit.readthedocs.io/) | Official docs | Complete toolkit | Active | Large |
| [DeepChem Tutorials](https://deepchem.readthedocs.io/) | Tutorial series | ML for chemistry | Active | Growing |
| [Qiskit Textbook](https://qiskit.org/textbook/) | Interactive book | Quantum computing | Active | Large |
| [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/) | Official docs | Graph neural networks | Active | Large |

---

## Quick Reference Links

### Essential Bookmarks

## Daily Use

- ChEMBL Database
- PubChem
- Protein Data Bank
- arXiv.org (cs.LG, q-bio.BM, quant-ph)
- Google Scholar

## Development Tools

- GitHub
- Jupyter Notebook
- Google Colab
- Kaggle
- Papers with Code

## Learning Resources

- Coursera
- edX
- YouTube - Educational Channels
- MIT OpenCourseWare
- Khan Academy

## Professional

- LinkedIn
- ORCID
- ResearchGate
- Conference Deadlines

---

*This resource collection is continuously updated. Bookmark this page and check for updates regularly. Contribute suggestions for new resources via GitHub issues or pull requests.*

**Last Updated**: December 2024
**Version**: 1.0
**Maintainer**: ChemML Project Team
