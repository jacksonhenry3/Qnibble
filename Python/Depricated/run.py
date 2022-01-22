from qnibblesOLD import *
"""
We should consider splitting in toa product of energy sub spaces, this should save memory 
by implicitly excluding the correltaitons between state in different energy subspace, 
which SHOULD be inaccessable to our methods)
"""
nqubits = 8
block_size = 2
num_blocks = nqubits // block_size
assert num_blocks == nqubits / block_size
# dm = n_thermal_qubits([1 if i != nqubits // 2 else 2 for i in range(nqubits)])
dm = n_thermal_qubits(np.random.rand(nqubits))
order = list(range(nqubits))
temphist = [temps(dm)]


for i in range(25):
    hi = [random_hamiltonian(block_size, block_size // 2) for i in range(num_blocks)]
    # see https://physics.stackexchange.com/questions/164109/dealing-with-tensor-products-in-an-exponent
    U = q.tensor(*[(-1j * .1 * h).expm() for h in hi])
    print(i)
    O = list(np.roll(order, np.random.randint(0, nqubits)))

    U = U.permute(O)
    dm = (U * dm * U.dag())
    # dm.tidyup()
    print(dm.data.count_nonzero())
    # temphist.append(temps(dm))
print('done')
plt.imshow(np.ceil(np.real(dm)), cmap='gray', interpolation='none')
plt.title(f"{nqubits} {block_size}")
# plt.imshow(np.real(dm), cmap='PuOr', interpolation='none')
plt.axis('off')
plt.tight_layout()
plt.show()
