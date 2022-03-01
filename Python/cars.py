from ket import Ket, Basis, canonical_basis, energy_basis

# basis = canonical_basis(3)
basis = energy_basis(10)
print(basis)

for b in basis:
    print(b)
