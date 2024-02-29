# Quantum Variational Circuit
In this directory is present a list of notebook containing our test regarding the classification of entangled states using QVC or other similar algorithm.
A VQC is usally composed of three different part: the first part is related to the data embedding, in other word how you load the features on your quantum circuit. There are different technics: Angel Embedding, Amplitude Embedding or Basis Embeddin. The second part is the proper variational circuit, it is usally composed as cnot and series of rotation in which the angle is a parameter to optimize. And finally the output of the circuit which is some type of measure.

## `qvc_haar_obs.ipynb`
This is out main test reagardin QVC, and the one which yelds better results. The dataset we are working with the dataset generated from the observables of a quantum state (`ds_haar_obs.csv`). 
The embedding of the data is done using the template `AngleEmbedding()` from pennylane in where it encodes $n$ features into the rotation angles of $n$ qubits. The template `StronglyEntangledLayers()` is used as our varational part of the circuit, and a exemple of the circuit for 3 features can be seen below:
<p align="center">
  <img src="https://github.com/Brusa99/EntanglementDetectionQML/blob/main/images/qvc.png">
</p>For the output of the circuit we measure the expectation values of $\sigma_z$ of the first wire.
The optimization algorithm used is Adam with the hinge loss as our cost function. In the other notebooks different optimization algorithms and cost function were used as: COBYLA and the cross entropy.

#### Other notebooks
Other test done using different algorithms are in the following notebooks:
- `qvc_mixed_obs.ipynb` A very similar notebook to the main one, this one uses a different dataset containing observable generated from mixed states. 
- `qvc_haar_stat.ipynb` In this notebook we implement an idea in which we load in the quantum circuit the state of an exemple and with a QVC we try to detect if the state is separable or not. The results are not great.
- `hybrid_qnn_obs.ipynb` A notebook implementing a hybrid neural network for the observables dataset. We didn't manage good results. 
- `qcnn.ipynb` An implementation of a quantum convolutional neural network for the observable dataset. Even here the results are good.

