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
	
def IGP(A, k):
	'''
    	Algorithm IGP from https://arxiv.org/pdf/1602.08529.pdf
    '''

    m, n = A.shape

    row_part = range(0, m+1, m/k)
    col_part = range(0, n+1, n/k)

    def column_update(i):
        C_n = np.arange(col_part[i-1], col_part[i])
        A_sub = submatrix(A, 
                 {'rows'   : ix['rows'],
                  'columns': C_n})

        new = np.argmax(A_sub.sum(axis = 0))

        ix['columns'] = np.append(ix['columns'], C_n[new]).astype('int')

    def row_update(i):
        R_n = np.arange(row_part[i-1], row_part[i])

        A_sub = submatrix(A,
                          {'rows'   : R_n,
                           'columns': ix['columns']})
        new = np.argmax(A_sub.sum(axis = 1))

        ix['rows'] = np.append(ix['rows'], R_n[new]).astype('int')

        i += 1
    
    
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
    
    return {'ix': ix, 
            'val': value(A, ix)}