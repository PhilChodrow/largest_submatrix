
import numpy as np

def read_data(path, clean_nan = False):
	m = np.loadtxt(path)
	m[:,0:2] = m[:,0:2] / (2.5*10**5)

	ix = m[:,0:2].astype('int')

	container = np.zeros((max(ix[:,0])+1, 
	                      max(ix[:,1])+1))

	for i in range(ix.shape[0]):
	    ix_i = ix[i,:]
	    container[ix_i[0], ix_i[1]] = m[i,2]

	if clean_nan:
		container = np.nan_to_num(container)
		
	return container
