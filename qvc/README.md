# Quantum variational classifiers
This directory contains various QVC algorithms that we used for the entanglement detection task.


- `Hybrid_QNN_obs.ipynb` : These notebook contains the implementation of an hybrid approach to Mmachine learning, 
  where a quantum variational layer is connected to some previous classical layer.
  Finally a classical output layer returns the outcome of the whole neural network.

  The results get from this appoach are unsuccessful, because we observe no increase in the accuracy during the various epochs,
  together with the abscence of a deacrease in the loss function.
