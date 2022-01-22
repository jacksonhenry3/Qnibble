"""#Examples"""
from qnibblesOLD import *
example_H = random_hamiltonian(4, 3)
# q.visualization.hinton(example_H)
h = example_H
i = q.identity(dims=[2, 2, 2, 2, 2])
eg = q.tensor(h, i) + q.tensor(i, h)
plt.imshow(np.absolute(np.absolute(eg.full())))
plt.axis('off')

fig = plt.imshow(np.absolute(np.absolute(example_H.full())))
plt.axis('off')

# initial density matrix
dm = n_thermal_qubits([100, 1, 10, 0])

q.visualization.hinton(dm)

for i in range(100):
    H = random_hamiltonian(4, 2)  # choose a random hamiltonian
    U = (-1j * H * 10).expm()  # generate the unitary evolution oeprator
    dm = U * dm * U.dag()  # update the dm

q.visualization.hinton(dm)

"""notice how the states with energy 0,1,3 or 4 are unchanged and developed no correlations."""
#
# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets
#
# H = random_hamiltonian(4,2) #choose a random hamiltonian
# U = (-1j*H*10).expm() #generate the unitary evolution oeprator
#
# def go(a=100,b=1,c=10,d=0):
#   #initial density matrix
#   dm = n_thermal_qubits([a,b,c,d])
#
#   # q.visualization.hinton(dm)
#
#
#   dm = U*dm*U.dag() #update the dm
#
#   q.visualization.hinton(dm)
#
# interact(go,a=(0,10),b=(0,10),c=(0,10),d=(0,10))

print(q.enr_thermal_dm([3, 1], 0, 100) == q.enr_thermal_dm([3, 1], 100, 100))

"""#more examples"""

dm = n_thermal_qubits([1, 1, 1, 1, 1, 1, 1, 1000])

dm





dm = n_thermal_qubits([10, 10, 10, 10, 100, 10, 10, 10])

order = list(range(8))
# plt.imshow(np.absolute(np.absolute(dm.full())))
# plt.axis('off')
# plt.show()
temphist = [temps(dm)]
for i in range(8):
    np.random.shuffle(order)
    hi1 = random_hamiltonian(4, 2)
    hi2 = random_hamiltonian(4, 2)
    H = q.tensor(hi1, I(4)) + q.tensor(I(4), hi2)
    U = (-1j * .5 * H).expm()
    dm = (U * dm.permute(order) * U.dag()).permute(inverse_order(order))
    temphist.append(temps(dm))

plt.imshow(np.real(temphist), cmap='coolwarm')
plt.axis('off')
plt.show()

# plt.imshow(np.absolute(np.absolute(dm.full())))
# plt.axis('off')
# plt.show()

nqubits = 12
block_size = 4
num_blocks = nqubits // block_size
assert num_blocks == nqubits / block_size
dm = n_thermal_qubits([1 if i != nqubits // 2 else 2 for i in range(nqubits)])
order = list(range(nqubits))
temphist = [temps(dm)]
for _ in range(12):
    print(_)
    O = list(np.roll(order, np.random.randint(0, nqubits)))
    IO = inverse_order(O)
    hi = [random_hamiltonian(block_size, block_size // 2) for i in range(num_blocks)]
    U = q.tensor(*[(-1j * .2 * h).expm() for h in
                   hi])  # see https://physics.stackexchange.com/questions/164109/dealing-with-tensor-products-in-an-exponent
    dm = dm.permute(O)
    dm = (U * dm * U.dag())
    dm = dm.permute(IO)
    temphist.append(temps(dm))

plt.imshow(np.real(temphist))
plt.axis('off')
plt.show()

h1 = n_thermal_qubits([1, 2])
h2 = n_thermal_qubits([3, 4])
H = q.tensor(h1, I(2)) + q.tensor(I(2), h2)

U1 = (H).expm()
U2 = q.tensor(h1.expm(), h2.expm())
assert U1 == U2

hi = [random_hamiltonian(block_size, block_size // 2) for i in range(num_blocks)]

blocks = [[block_size * n + i for i in range(block_size)] for n in range(num_blocks)]

# standard
U = q.tensor(*[(-1j * .2 * h).expm() for h in hi])
res1 = (U * dm * U.dag())

# experimental
temp_dm = dm.copy()
# for i,block in enumerate(blocks):
#   block_dm = q.ptrace(dm,block)
#   temp_dm-=q.tensor([I(len(b)) if b!=block else block_dm for b in blocks])
#   Ui = (-1j*.2*hi[i]).expm()
#   block_dm = (Ui*block_dm*Ui.dag())
#   temp_dm += (q.tensor([I(len(b)) if b!=block else block_dm for b in blocks]))


for i, block in enumerate(blocks):
    print(block)
    #

q.visualization.hinton(q.ptrace(res1, [0, 1]))

# q.visualization.hinton(dm)

