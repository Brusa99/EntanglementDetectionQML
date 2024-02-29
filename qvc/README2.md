# Quantum Variational Circuit
In this directory is present a list of notebook containing our test regarding the classification of entangled states using QVC or other similar algorithm.
A VQC is usally composed of two different part: the first part is related to the data embedding, in other word how you load the features on your quantum circuit. There are different technics: Angel Embedding, Amplitude Embedding or Basis Embeddin. The second part is the proper variational circuit, it is usally composed as cnot and series of rotation in which the angle is a parameter to optimize.

## `qvc_haar_obs.ipynb`
This is out main test reagardin QVC, and the one which yelds better results. The dataset we are working with the dataset generated from the observables of a quantum state (`ds_haar_obs.csv`). 
The embedding of the data is done using the template `AngleEmbedding()` from pennylane in where it encodes $n$ features into the rotation angles of $n$ qubits. The template `StronglyEntangledLayers()` is used as our varational part of the circuit, and a exemple of the full circuit can be seen in the figure below:



#### Other notebooks
Other test done using different algorithms are in the following notebooks:
- `qvc_mixed_obs.ipynb` A very similar notebook to the main one, this one uses a different dataset containing observable generated from mixed states. 
- `qvc_haar_stat.ipynb` In this notebook we implement an idea in which we load in the quantum circuit the state of an exemple and with a QVC we try to detect if the state is separable or not. 
- `Hybrid_QNN_obs.ipynb` 

