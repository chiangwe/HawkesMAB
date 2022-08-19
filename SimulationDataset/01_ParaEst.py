import numpy as np
import pdb
import pandas as pd
import math
import itertools
import pickle
import copy
from tqdm import tqdm
import Hawkes as hk

# Boundary Detection 
def bound_proj( value, limits):
	if value > limits[1]:
		value = limits[1]
	if value < limits[0]:
		value = limits[0]
	return value

# Load Data 
with open("./data/raw_data.pkl", "rb") as fp:
	TsSet = pickle.load( fp)

# Load Timestamp
N_Ts = np.load('./data/N_Ts.npy')

# Load parameter
with open("./data/paras.pkl", "rb") as fp:
	paras = pickle.load( fp)

globals().update(paras)
print("paras: ", paras)

# Get prior variables estimation 
arr_para_prior = np.zeros( (4, Ktol) )

for itr_ts in (range( 0, Ktol)):
	
	# Estimate the parameter
	model = hk.estimator().set_kernel('exp',num_exp=1).set_baseline('const')
	
	itv_est = [ 0, rounds_tr*step_size ]
	opt = {'stop': 10000}
	
	all_ts = TsSet[itr_ts][0:rounds_tr]
	all_ts = list(itertools.chain(*all_ts))
	
	model.fit( all_ts, itv_est, prior=[],opt=opt)
	
	dict_est       = copy.deepcopy( model.parameter)
	dict_est['R0'] = copy.deepcopy( dict_est['alpha'] )
	dict_est['alpha'] = dict_est['R0']*dict_est['beta']
	
	arr_para_prior[:, itr_ts] = np.array([dict_est['mu'],  dict_est['R0'], dict_est['beta'], dict_est['alpha']])
	
# Estimate in the total training period
# Estimate the parameters 
arr_para = np.zeros( (4, Ktol, tol_rounds) )
est_bool = np.zeros( (   Ktol, tol_rounds) )

for itr_ts in (range( 0, Ktol)):
	
	for rounds in tqdm(range( 0, tol_rounds )):
		
		# Estimate the parameter
		model = hk.estimator().set_kernel('exp',num_exp=1).set_baseline('const')

		itv_est = [ step_size*rounds, step_size*(rounds+1)]
		
		if len(TsSet[itr_ts][rounds]) > 0:
			est_bool[itr_ts, rounds] = 1
			
		try:

			opt = {'stop': 10000}

			model.fit( TsSet[itr_ts][rounds], itv_est, prior=[],opt=opt)

			dict_est       = copy.deepcopy( model.parameter)
			dict_est['R0'] = copy.deepcopy( dict_est['alpha'] )
			dict_est['alpha'] = dict_est['R0']*dict_est['beta']
			
			if dict_est['R0'] < 1:
				arr_para[:, itr_ts, rounds] = \
					np.array([dict_est['mu'],  dict_est['R0'], dict_est['beta'], dict_est['alpha']])
			else:
				arr_para[:, itr_ts, rounds] = arr_para_prior[:, itr_ts]
			
			if np.isnan(np.array([dict_est['mu'],  dict_est['R0'], dict_est['beta'], dict_est['alpha']])).any():
				arr_para[:, itr_ts, rounds] = arr_para_prior[:, itr_ts]
		except:
			print("Fail")
			arr_para[:, itr_ts, rounds] = \
					np.array([np.mean(mu_prior), np.mean(R0_prior), np.mean(beta_prior), np.mean(alpha_prior) ])

np.savez("./data/para_est", arr_para=arr_para, est_bool=est_bool)	
#pdb.set_trace()

