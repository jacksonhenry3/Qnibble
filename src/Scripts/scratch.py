from src import density_matrix as DM

pops = [.1, .2]
sys = DM.nqbit(pops)
data = sys.data.toarray()
print(data @ data)
test = sys.qbit_basis()

print(test)
