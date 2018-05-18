import numpy as np
from docplex.mp.model import Model
import utils

def int_programming(data,k):
    m,n = data.shape
    print(m,n)

    # Create CPLEX model
    mdl = Model(name='IP')

    # Create m*n + m + n binary decision variables for every matrix entry and every row and column respectively
    x = mdl.binary_var_list(m*n,name="X")
    R = mdl.binary_var_list(m,name="R")
    C = mdl.binary_var_list(n,name="C")

    # Add the constaints that only k rows and columns can be selected
    s_R = mdl.sum(R)
    s_C = mdl.sum(C)
    mdl.add_constraint(s_R==k)
    mdl.add_constraint(s_C==k)

    # Enforce that the selected matrix entries correspond to the selected rows and columns
    row_sums = [sum(x[i*n:i*n+n]) for i in range(m)]
    col_sums = [sum([x[n*i+j] for i in range(m)]) for j in range(n)]
    row_constraints = [row_sums[i] == k*R[i] for i in range(m)]
    col_constraints = [col_sums[i] == k*C[i] for i in range(n)]
    mdl.add_constraints(row_constraints)
    mdl.add_constraints(col_constraints)

    # Maximize the size of the submatrix
    obj = mdl.sum([x[n*i+j]*data[i,j] for i in range(m) for j in range(n)])
    mdl.maximize(obj)

    # Solve the integer programming and print results
    sol = mdl.solve()
    mdl.print_solution()
    mdl.report_kpis()

    return sol

''' 
Example Usage
d = utils.read_data('chr1_chr2.txt')
d = np.asarray(d)
sol = int_programming(d,9)
'''
