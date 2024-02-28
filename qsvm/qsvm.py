import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import svm

from qiskit.circuit.library import PauliFeatureMap
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_algorithms.state_fidelities import ComputeUncompute, BaseStateFidelity
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from typing import Optional, Tuple


def get_feature_map(n_features: int,
                    paulis: list[str] = None,
                    reps: int = 2,
                    entanglement: str = 'linear') -> QuantumCircuit:
    """Wrapper to return a feature map quantum circuit.

    The function returns a quantum circuit that maps the input data to a quantum state.
    By default, the feature map is a [Z, ZZ] PauliFeatureMap with linear entanglement.

    Args:
        n_features (int): feature dimension.
        paulis (list[str]): list of Pauli gates to use in the feature map.
        reps (int): number of repetitions of the feature map.
        entanglement (str): entanglement strategy for the feature map.

    Returns:
        QuantumCircuit: the feature map quantum circuit.
    """
    # set mutable defaults
    if paulis is None:
        paulis = ['Z', 'ZZ']

    # get feature map
    feature_map = PauliFeatureMap(feature_dimension=n_features, reps=reps, entanglement=entanglement, paulis=paulis)
    return feature_map


# kernel functions

def manual_quantum_kernel(x1, x2, feature_map=None, **kwargs) -> float:
    """Compute the quantum kernel between two input vectors using numpy.

    The function computes the quantum kernel between two input vectors using the provided feature map. If no feature map
    is provided, the default feature map obtained by the `get_feature_map` is used.

    The kernel product is computed as the squared absolute value of the braket product between the statevectors

    Args:
        x1: first input vector.
        x2: second input vector.
        feature_map: feature map to use for encoding the input vectors.
        **kwargs: extra unused arguments.

    Returns:
        float: the kernel product between the two input vectors.
    """
    # check input and set defaults
    assert len(x1) == len(x2), "Input vectors must have the same length"
    if feature_map is None:
        feature_map = get_feature_map(len(x1))

    # encode inputs with the feature map
    qc1 = feature_map.assign_parameters(x1)
    qc2 = feature_map.assign_parameters(x2)
    # compute the statevectors
    sv1 = Statevector.from_instruction(qc1).data
    sv2 = Statevector.from_instruction(qc2).data
    # compute the kernel product
    return np.abs(sv1.conjugate().dot(sv2)) ** 2


def qiskit_quantum_kernel(x1, x2,
                          fidelity: Optional[BaseStateFidelity] = None,
                          feature_map: Optional[PauliFeatureMap] = None,
                          **kwargs, ) -> float:
    """Compute the quantum kernel between two input vectors using Qiskit's FidelityQuantumKernel class.

    The function computes the quantum kernel between two input vectors using the provided feature map. If no feature map
    is provided, the default feature map obtained by the `get_feature_map` is used.

    If fidelity is not provided, ComputeUncompute is used.

    Args:
        x1: first input vector.
        x2: second input vector.
        fidelity: An instance of the ~qiskit_algorithms.state_fidelities.BaseStateFidelity primitive to be used to
         compute fidelity between states.
        feature_map: feature map to use for encoding the input vectors.
        **kwargs: extra unused arguments.

    Returns:
        float: the kernel product between the two input vectors.
    """
    # check input and set defaults
    assert len(x1) == len(x2), "Input vectors must have the same length"
    if feature_map is None:
        feature_map = get_feature_map(len(x1))
    n = len(x1)

    kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    kernel_value = kernel.evaluate(x1, x2)
    return kernel_value[0, 0]


def sampler_quantum_kernel(x1, x2,
                           sampler: Optional[Sampler] = None,
                           feature_map: Optional[PauliFeatureMap] = None,
                           shots: int = 1024,
                           **kwargs, ) -> float:
    """Compute the quantum kernel between two input vectors using a quantum sampler.

    The function computes the quantum kernel between two input vectors using the provided feature map. If no feature map
    is provided, the default feature map obtained by the `get_feature_map` is used.

    If sampler is not provided, a sampler is instanciated. This should be avoided for efficiency.

    The kernel product is defined by the probability of measuring 0^otimes n in the computational basis.


    Args:
        x1: first input vector.
        x2: second input vector.
        sampler: quantum sampler to use for sampling the measures.
        feature_map: feature map to use for encoding the input vectors.
        shots: number of shots to use for sampling the measures.
        **kwargs: extra unused arguments.

    Returns:
        float: the kernel product between the two input vectors.
    """
    # check input and set defaults
    assert len(x1) == len(x2), "Input vectors must have the same length"
    if feature_map is None:
        feature_map = get_feature_map(len(x1))
    if sampler is None:
        sampler = Sampler()
    n = len(x1)

    # encode inputs with the feature map
    x1_enc = feature_map.assign_parameters(x1)
    x2_enc = feature_map.assign_parameters(x2)
    # construct the kernel circuit
    qc = QuantumCircuit(n, n)
    qc.append(x1_enc, range(n))
    qc.append(x2_enc, range(n))
    qc.measure(range(n), range(n))
    # sample the kernel circuit
    result = sampler.run(qc, shots=shots).result()
    # compute the kernel value
    # kernel_value = result.quasi_dists[0].get(0, 0) / shots
    kernel_value = result.quasi_dists[0].binary_probabilities()
    # get list of dict keys
    states = list(kernel_value.keys())
    # prob of measuring all 0s
    return kernel_value[states[0]]


# training functions

def train(X: pd.DataFrame,
          y: pd.Series,
          feature_map: Optional[PauliFeatureMap] = None,
          kernel_function: Optional[str] = 'manual',
          sampler: Optional[Sampler] = None,
          fidelity: Optional[BaseStateFidelity] = None, ) -> svm.SVC:
    """Train the QSVM model using the input data and the feature map.

    The function computes the kernel matrix using the input data, the feature map and the selected kernel.
    The kernel matrix is then used to train a classical SVM model.

    The kernel function can be 'manual', 'sampler' or 'qiskit'.
    For 'sampler' a sampler can be provided. For 'qiskit' a fidelity can be provided.

    Args:
        X: Input training data.
        y: Labels for the training data.
        feature_map: feature map to use for encoding the input vectors.
        kernel_function: kernel function to use for computing the kernel matrix. Possible values are 'manual', 'sampler'
         and 'qiskit'.
        sampler: quantum sampler to use for sampling the measures. Only used if kernel_function is 'sampler'.
        fidelity: fidelity to use for computing the kernel matrix. Only used if kernel_function is 'qiskit'.

    Returns:
        svm.SVC: the trained SVM model.
    """

    # check inputs and set defaults
    assert kernel_function in ['manual', 'sampler', 'qiskit'], "Invalid kernel function"
    if feature_map is None:
        feature_map = get_feature_map(X.shape[1])
    if kernel_function == 'sampler' and sampler is None:
        sampler = Sampler()
    if kernel_function == 'qiskit' and fidelity is None:
        if sampler is None:
            sampler = Sampler()
        fidelity = ComputeUncompute(sampler=sampler)
    # get the kernel function
    if kernel_function == 'manual':
        kernel_function = manual_quantum_kernel
    elif kernel_function == 'sampler':
        kernel_function = sampler_quantum_kernel
    elif kernel_function == 'qiskit':
        kernel_function = qiskit_quantum_kernel

    # compute the kernel matrix
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = kernel_function(X.iloc[i], X.iloc[j], sampler=sampler, feature_map=feature_map, fidelity=fidelity)
            K[j, i] = K[i, j]  # the kernel matrix is symmetric

    # train the SVM model
    model = svm.SVC(kernel='precomputed')
    model.fit(K, y)
    return model


def train_test(X_train: pd.DataFrame,
               y_train: pd.Series,
               X_test: pd.DataFrame,
               y_test: pd.Series,
               feature_map: Optional[PauliFeatureMap] = None,
               kernel_function: Optional[str] = 'manual',
               sampler: Optional[Sampler] = None,
               fidelity: Optional[BaseStateFidelity] = None,
               ) -> Tuple[svm.SVC, float]:
    """Wrapper to train the QSVM model and test it on unseen data.

    The function trains the QSVM model using the input training data and then tests it on the input test data.
    The function returns the trained model and the test score (accuracy).

    Args:
        X_train: Input training data.
        y_train: Labels for the training data.
        X_test: Test data.
        y_test: Labels for the test data.
        feature_map: feature map to use for encoding the input vectors.
        kernel_function: kernel function to use for computing the kernel matrix. Possible values are 'manual', 'sampler'
         and 'qiskit'.
        sampler: quantum sampler to use for sampling the measures. Only used if kernel_function is 'sampler'.
        fidelity: fidelity to use for computing the kernel matrix. Only used if kernel_function is 'qiskit'.

    Returns:
        Tuple[svm.SVC, float]: the trained SVM model and the test score (accuracy).

    """
    model = train(X_train,
                  y_train,
                  feature_map=feature_map,
                  kernel_function=kernel_function,
                  sampler=sampler,
                  fidelity=fidelity)

    # get the kernel function
    if kernel_function == 'manual':
        kernel_function = manual_quantum_kernel
    elif kernel_function == 'sampler':
        kernel_function = sampler_quantum_kernel
    elif kernel_function == 'qiskit':
        kernel_function = qiskit_quantum_kernel

    # compute the kernel matrix for the test data
    n = len(X_test)
    m = len(X_train)
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K[i, j] = kernel_function(X_train.iloc[j],
                                      X_test.iloc[i],
                                      sampler=sampler,
                                      feature_map=feature_map,
                                      fidelity=fidelity)

    # test the SVM model
    score = model.score(K, y_test.values)

    # return the trained model
    return model, score


# plot functions

def plot_results(features, times, accuracy):
    # plot on the same figure the time and the accuracy
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Time (s)', color=color)
    ax1.plot(features, times, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)  # we already handled the x-label with ax1
    ax2.plot(features, accuracy, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

