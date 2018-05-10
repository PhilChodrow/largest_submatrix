import numpy as np
from scipy.optimize import minimize, Bounds

def submatrix(A, ix):
	'''
	Get the submatrix corresponding to a given set of row and column indices. 
	A: a 2d np.array
	ix: dict of the form {'rows'    : np.array([1, 2, 3, 4]),
						  'columns' : np.array([1, 2, 3, 4])}
	'''
	return A[ix['rows'][:, np.newaxis], ix['columns']]
	
def value(A, ix):
	'''
	Get the value (average) of a submatrix corresponding to a given set of row 
	and column indices. 
	A: a 2d np.array
	ix: dict of the form {'rows'    : np.array([1, 2, 3, 4]),
						  'columns' : np.array([1, 2, 3, 4])}
	'''
	return np.mean(submatrix(A, ix))

def inclusion_matrix(A, ix):

	m, n = A.shape

	x = np.zeros(m)
	y = np.zeros(n)

	x[ix['rows']] = 1
	y[ix['columns']] = 1

	return np.outer(x,y)


def LAS(A, k):
	'''
		Algorithm LAS from https://arxiv.org/pdf/1602.08529.pdf
		A: a 2d np.array
		k: a positive integer
	'''
	m, n = A.shape

	# given a fixed set of row indices, find the k columns whose sums 
	# in those indices are largest. 
	def column_update():
		sums = A[ix['rows'],:].sum(axis = 0)     # restrict to given row indices
		new_ix = np.argpartition(-sums, k)[0:k]  # find k largest restricted cols
		ix['columns'] = new_ix                   # replace old indices with new

	# given a fixed set of column indices, find the k rows whose sums 
	# in those indices are largest. 
	def row_update():
		sums = A[:, ix['columns']].sum(axis = 1) # restrict to given column indices
		new_ix = np.argpartition(-sums, k)[0:k]  # find the k largest restricted rows
		ix['rows'] = new_ix                      # replace old indices with new

	# randomly initialize indices
	ix = {'rows'    : np.random.choice(m, k, replace=False),
		  'columns' : np.random.choice(n, k, replace=False)}    
	
	# initialize loop parameters. 
	prev_val = -10**(8)
	val = 0
	eps = 10**(-8) # tolerance: when improvement in val is less than this, stop.
  
	# main loop
	while np.abs(val - prev_val) > eps:
		prev_val = val
		row_update()
		column_update()
		val = value(A, ix)
	
	# return result as dict, including the indices and the value. 
	return({'ix' : ix, 'val': val})
	
def IGP(A, k):
	'''
		Algorithm IGP from https://arxiv.org/pdf/1602.08529.pdf
		A: a 2d np.array
		k: a positive integer
	'''

	m, n = A.shape

	# partition the row and column indices into k+1 approximately-equal segments 
	# R_i and C_i for i in 1...k+1
	row_part = range(0, m+1, m/k)
	col_part = range(0, n+1, n/k)

	# fixing current row indices, search through the columns within C_i the the one
	# that most increases the value. 
	def column_update(i):
		C_i = np.arange(col_part[i-1], col_part[i]) # grab C_i
		A_sub = submatrix(A, 						# restrict to columns in C_i
				 {'rows'   : ix['rows'],            # and rows in ix
				  'columns': C_i})

		new = np.argmax(A_sub.sum(axis = 0))        # index of best addition in C_i

		# add best addition to ix
		ix['columns'] = np.append(ix['columns'], 
								  C_i[new]).astype('int')

	def row_update(i):
		R_i = np.arange(row_part[i-1], row_part[i])   # grab R_i

		A_sub = submatrix(A,                          # restrict to rows in R_i
						  {'rows'   : R_i,            # and columns in ix
						   'columns': ix['columns']})
		new = np.argmax(A_sub.sum(axis = 1))          # index of best addition in R_i

		# add best addition to ix
		ix['rows'] = np.append(ix['rows'], 
							   R_i[new]).astype('int')

	# initialize
	ix = {'rows'    : np.random.choice(row_part[1], 1),
		  'columns' : []}     
	column_update(1)

	i = 2
	
	# main loop
	while i <= k:
		row_update(i)
		column_update(i)
		i += 1
	
	# output as dict including indices and value
	return {'ix': ix, 
			'val': value(A, ix)}

def round_off(x, k):
    x_ind = np.argpartition(x, -k)[-k:]
    x = np.zeros(x.shape)
    x[x_ind] = 1
    return x

def continuous_SLSQP(A, k, print_obj = False):
	m, n = A.shape

	# objective function
	def f(z): 
	    val = np.dot(np.dot(z[:m], A, ), z[m:]) / (k**2)
	    if print_obj:
	    	print val
	    return -val

	# objective gradient
	def jac(z):
	    return -np.concatenate((np.dot(A, z[m:]), np.dot(z[:m], A))) / (k**2)

	# constraints
	e_x = np.concatenate((np.ones(m), np.zeros(n)))
	e_y = np.concatenate((np.zeros(m), np.ones(n)))

	bnds = Bounds(0, 1)
	cons = ({'type': 'ineq', 'fun': lambda z: k - z[:m].sum(), 'jac' : lambda z: -e_x},
	        {'type': 'ineq', 'fun': lambda z: k - z[m:].sum(), 'jac' : lambda z: -e_y})

	# random start

	x_0 = np.random.rand(m)
	y_0 = np.random.rand(n)
	x_0 = k*x_0 / x_0.sum()
	y_0 = k*y_0 / y_0.sum()
	z_0 = np.concatenate((x_0, y_0))

	# perform optimization
	res = minimize(f, 
	               z_0, 
	               method='SLSQP', 
	               bounds = bnds,
	               constraints = cons, 
	               jac = jac, 
	               options = {'disp': True})

	z = res['x']
	x = z[:m]
	y = z[m:]

	x = round_off(x, k)
	y = round_off(y, k)

	return {'x' : x, 'y' : y, 'val' : np.dot(np.dot(x, A), y) / (k**2)}











