import numpy as np
from src import (
    density_matrix as DensityMatrix,
    random_unitary)  # Replace with the actual import if needed
from src.random_unitary import random_energy_preserving_unitary  # Replace if needed

import numpy as np
import matplotlib.pyplot as plt

class Ket:
    def __init__(self, num, num_qbit):
        self.num = num
        self.num_qbit = num_qbit

    def __repr__(self):
        return f"Ket(num={self.num}, num_qbit={self.num_qbit})"

    def reorder(self, order):
        # Validate the order list
        if len(order) != self.num_qbit or sorted(order) != list(range(self.num_qbit)):
            raise ValueError("Order list must contain all indices from 0 to num_qbit-1")

        # Convert number to binary string and pad to the correct bit length
        binA = bin(self.num)[2:].zfill(self.num_qbit)

        # Reorder the bits according to the 'order' list
        reordered_bin = ''.join(binA[i] for i in order)

        # Convert the reordered binary string back to an integer and return a new Ket object
        return Ket(int(reordered_bin, 2), self.num_qbit)


def reconstruct_dense(dm):
    # Infer the shape if not directly available
    if hasattr(dm, "shape"):
        shape = dm.shape  # Use shape attribute if available
    else:
        max_row = max([coord[0] for coord in dm.data.keys()])
        max_col = max([coord[1] for coord in dm.data.keys()])
        shape = (max_row + 1, max_col + 1)  # Add 1 since indices are 0-based

    # Create an empty dense matrix
    dense_matrix = np.zeros(shape)

    # Populate the matrix with values from dm.data
    for (row, col), value in dm.data.items():
        dense_matrix[row, col] = value

    return dense_matrix


import numpy as np

import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

def relabel_matrix_entries(matrix, n, new_order):
    """
    Relabel the rows and columns of a matrix according to the new qubit order.

    Parameters:
    - matrix: Input matrix (2^n x 2^n) for an n-qubit system.
    - n: Number of qubits.
    - new_order: List defining the new qubit order (1-based indexing).

    Returns:
    - new_labels: The new computational basis labels after the relabeling.
    """
    dim = 2 ** n

    # Generate computational basis states (row/column labels)
    basis_states = [format(i, f'0{n}b') for i in range(dim)]

    # Compute the permutation indices based on the new_order
    permutation = []
    for state in basis_states:
        new_state = ''.join(state[new_order.index(i + 1)] for i in range(n))
        permuted_index = int(new_state, 2)
        permutation.append(permuted_index)

    # Generate the new labels based on the new ordering
    new_labels = [basis_states[i] for i in permutation]

    return new_labels


def plot_matrix_with_labels(matrix, new_labels):
    """
    Plot the matrix with new computational basis labels as axes.

    Parameters:
    - matrix: The matrix to plot.
    - new_labels: The new computational basis labels (after relabeling).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(np.abs(matrix), cmap='viridis')

    # Set the labels for the rows and columns (no change in the matrix)
    ax.set_xticks(np.arange(len(new_labels)))
    ax.set_yticks(np.arange(len(new_labels)))
    ax.set_xticklabels(new_labels, rotation=90)
    ax.set_yticklabels(new_labels)

    # Add color bar and title
    plt.colorbar(cax)
    plt.title("Matrix with New Computational Basis Labels")

    plt.show()




num_chunks =2
chunk_size=2
basis = DensityMatrix.energy_basis(chunk_size)
identity = DensityMatrix.Identity(basis)
sub_unitary = random_unitary.haar_random_unitary(theta_divisor=1, phi_divisor=1, omega_divisor=1, seed=None)

composite_unitaries = [DensityMatrix.tensor([sub_unitary if i == j else identity for i in range(num_chunks)]) for j in
                           range(num_chunks)]
unitary = np.prod(composite_unitaries)

# Example usage:
N = 4
matrix = unitary.data.toarray() # Example random 3-qubit matrix
new_order = [1, 4, 2, 3]  # Reorder: qubit 1->1, qubit 2->3, qubit 3->2
# Relabel the matrix
new_labels = relabel_matrix_entries(matrix, N, new_order)

# Plot the matrix with new labels (structure of the matrix stays the same)
plot_matrix_with_labels(matrix, new_labels)








#print("Original Matrix:")
#print(matrix)
#print("\nRelabeled Matrix:")
#print(relabeled_matrix)

def plot_matrix(matrix, title="Matrix Plot"):
    """
    Plot a 2D matrix as a heatmap.

    Parameters:
    matrix (np.ndarray): The 2D matrix to be plotted.
    title (str): The title of the plot.
    """
    plt.figure(figsize=(8, 8))  # Adjust the figure size if necessary
    plt.imshow(np.abs(matrix), cmap="viridis", interpolation="nearest")  # Use a color map for visualization
    plt.colorbar(label="Matrix Value")  # Add a colorbar to indicate the values
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.grid(False)  # Optional: Disable grid lines for a cleaner look
    plt.show()

plot_matrix(matrix)

#plot_matrix(relabeled_matrix)

# Example: Simulating a simple DensityMatrix and Unitary
def test_relabel_basis():
    # Mock density matrix and unitary matrix for debugging
    num_qubits = 8  # Small system for debugging
    ket = Ket(0b01101, 5)
    print(ket.reorder([2, 0, 4, 1, 3]))
    dm = DensityMatrix.n_thermal_qbits([0.1,0.2,0.15,0.3])  # Replace with actual initialization if different

#unitary = random_energy_preserving_unitary(num_qubits)
    num_chunks =2
    chunk_size=2
    basis = DensityMatrix.energy_basis(chunk_size)
    identity = DensityMatrix.Identity(basis)
    sub_unitary = random_unitary.haar_random_unitary(theta_divisor=1, phi_divisor=1, omega_divisor=1, seed=None)

    composite_unitaries = [DensityMatrix.tensor([sub_unitary if i == j else identity for i in range(num_chunks)]) for j in
                           range(num_chunks)]
    unitary = np.prod(composite_unitaries)
    # Define order for relabeling
    order = [[0,3],[1,2]]  # Simple swap for a 2-qubit system
    print(f"Original Order: {order}")

    # BEGIN Debugging
    print("=== Step 1: Initial Matrices ===")
    print("Density Matrix (dm):")
    dm.plot()
    print(dm.plot)  # Show the initial density matrix
    print("\nUnitary Matrix (unitary):")
    print(unitary.basis)
    unitary.plot()

    order = [qbit for chunk in order for qbit in chunk]
    print(f"Flat Order: {order}")

    assert set(list(order)) == set(range(dm.number_of_qbits)), f"{set(order)} vs {set(range(dm.number_of_qbits))}"
    # 1. Relabel basis
    print("\n=== Step 2a: Relabeling Basis ===")
    try:
        print("Relabeling unitary basis...")
        unitary.relabel_basis(order)
        print("Unitary after relabeling:")
        print(unitary.basis)
        unitary.plot()
    except Exception as e:
        print(f"Error during relabel_basis: {e}")

    # 2. Change to energy basis
    print("\n=== Step 3: Changing to Energy Basis ===")
    try:
        print("Changing unitary to energy basis...")
        unitary.change_to_energy_basis()
        print(unitary.data)
        unitary.plot()
        print("Changing density matrix to energy basis...")
        dm.change_to_energy_basis()
        print(dm.data)
        dm.plot()
    except Exception as e:
        print(f"Error during change_to_energy_basis: {e}")

    # 3. Matrix Multiplication
    print("\n=== Step 4: Matrix Multiplication ===")
    try:
        print("Calculating dm = Unitary * dm * Unitary.H...")
        evolved_dm = unitary * dm * unitary.H
        print("Evolved Density Matrix:")
        print(evolved_dm.data)
    except Exception as e:
        print(f"Error during matrix multiplication: {e}")

    # 4. Verify matrix properties
    print("\n=== Step 5: Verifications ===")
    try:
        print("Checking that the density matrix is Hermitian...")
        assert np.allclose(evolved_dm, evolved_dm.H), "Error: Evolved density matrix is not Hermitian!"

        print("Checking trace equals 1...")
        assert np.isclose(np.trace(evolved_dm),
                          1), f"Error: Trace of evolved density matrix is not 1 (trace={np.trace(evolved_dm)})"

        print("Checking eigenvalues are non-negative...")
        eigenvalues = np.linalg.eigvals(evolved_dm)
        assert np.all(eigenvalues >= 0), f"Negative eigenvalues found in evolved density matrix: {eigenvalues}"

        print("All checks passed!")
    except Exception as e:
        print(f"Verification failed: {e}")


# Run the test script
if __name__ == "__main__":
    test_relabel_basis()
