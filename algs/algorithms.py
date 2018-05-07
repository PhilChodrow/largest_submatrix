import numpy as np

def greedy_search(k, method = 'LAS'):
	'''
		init should also include intersection of row/column with largest sums
		method in {'LAS', 'IGP'}
		Methods from https://arxiv.org/pdf/1602.08529.pdf
	'''
	if method == 'LAS':
		return LAS_search(k)
	else: 
		return IGP_search(k)

	print 'not implemented'

def submatrix(A, ix):
    return A[ix['rows'][:, np.newaxis], ix['columns']]

def value(A, ix):
    return np.mean(submatrix(A, ix))

def LAS(A, k, n_iters = 1):
    '''
    	Algorithm LAS from https://arxiv.org/pdf/1602.08529.pdf
    '''
    m, n = A.shape

    def column_update():
        sums = A[ix['rows'],:].sum(axis = 0)
        new_ix = np.argpartition(-sums, k)[0:k]
        ix['columns'] = new_ix

    def row_update():
        sums = A[:, ix['columns']].sum(axis = 1)
        new_ix = np.argpartition(-sums, k)[0:k]
        ix['rows'] = new_ix

    # initialize
    ix = {'rows'    : np.random.choice(m, k, replace=False),
          'columns' : np.random.choice(n, k, replace=False)}    
    
    prev_val = -10**(8)
    val = 0
    eps = 10**(-8)
  
    # main loop
    while np.abs(val - prev_val) > eps:
        prev_val = val
        row_update()
        column_update()
        val = value(A, ix)
    
    # return result as dict
    return({'ix' : ix, 'val': val})
	

def IGP_search(k):
	print 'not implemented'


def IP(A, m = None, n = None):
	print 'not implemented'
	