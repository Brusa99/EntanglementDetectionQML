# EntanglementDetectionQML ⚛🤖

Collection of notebooks for experiments in the use of (Quantum) Machine Learning for entanglement detection.  
This repository is part of the project for the exam of _Advanced Quantum Mechanics_ @ units 2023/24.

## Authors

- [Simone Brusatin](https://github.com/Brusa99) <[simone.brusatin@studenti.units.it](mailto:simone.brusatin@studenti.units.it)>  
- [Christian Candeago](https://github.com/oooidw) <[christian.candeago@studenti.units.it](mailto:CHRISTIAN.CANDEAGO@studenti.units.it)>  
- [Alessio D'Anna](https://github.com/Alessio-DAnna) <[alessiomaria.d'anna@studenti.units.it](mailto:ALESSIOMARIA.D'ANNA@studenti.units.it)>  
- [Paolo Da Rold](https://github.com/paolodr98) <[paolo.darold@studenti.units.it](mailto:PAOLO.DAROLD@studenti.units.it)>


## Contents

The following is a brief description of the subdirectories of the project.
More detailed information is contained in each subdirectory `README.md`.

- `datasets/` Contains the datasets used for the project and the notebook to generate them.
- `qsvm/` Contains various notebooks that use _Quantum Support Vector Machines_ to classify the data.
- `qvc/` Contains notebooks using _Quantum Variational Circuits_ like _Quantum Neural Networks_, _HybridNNs_Classifiers and _Quantum Convolutional NNs_.
- `classic/` Contains notebook that use non-quantum ML techniques.
- `images/` Contains images for the readmes.


## Goal and Motivation

The goal of the project is classify separable and entangled quantum states using (Quantum) Machine Learning techniques.

Entanglement is an essential resource for quantum information processing tasks, but finding a robust and _efficient_ method for detecting entanglement is still an open problem.
Various criterion exist to determine if a state is entangled, but these criterions require full quantum state tomography followed by density matrix estimation.
In particular, we aim to create an entanglement approach which maximizes accuracy while minimizing the number of measurements of the system.


## Brief Description

More detailed descriptions of each algorithm and the dataset generation are found in each subdirectories `README.md`.

Various datasets are generated that cover pure and mixed states.
From the previous datasets, observables are applied to obtain new datasets.
Various machine learning algorithms are then applied to classify the data.
