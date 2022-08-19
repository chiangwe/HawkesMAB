import sys
sys.path.append('../')
import os

import numpy as np
import pickle

import pdb
import itertools
import pdb
import sys


# Get the parameters 

def hyper_para( method, sim_phase ):
	
	# Load parameter
	with open("./data/paras.pkl", "rb") as fp:
		paras = pickle.load( fp)
	globals().update( paras )
	
	if (sim_phase == "val_phase_cal"):
		all_rounds = rounds_tr
		save_str = 'val_results'
		
	elif(sim_phase == "te_phase_cal"):
		all_rounds = tol_rounds - rounds_tr
		save_str = 'results'
		
	# Go through all files 
	all_filenames = os.listdir('./' + save_str + '/' + method + '/')
	
	# String parse parameters

	dict_para_tol = {}
	
	parse_str = [ filename.replace('.npz','') for filename in all_filenames ]
	n_para = int( len(parse_str[0].split('_'))/2 ) - 2
	name_para = [ parse_str[0].split('_')[2*itr_para+4] for itr_para in range(0, n_para) ]
	
	ls_K = list( set([ each.split('_')[1] for each in parse_str ]) )
	ls_K.sort()
	
	dict_para = dict( zip( ['K_' + each for each in ls_K ],  range(0, len(ls_K) ))  )
	dict_para_tol.update({'K':dict_para})
	
	ls_sim = list( set([ each.split('_')[3] for each in parse_str ]) )
	ls_sim.sort()
	
	dict_para = dict( zip( ['sim_' + each for each in ls_sim],  range(0, len(ls_sim) ))  )
	dict_para_tol.update({'sim':dict_para})
	
	for itr_para in range(0, n_para):
		ls_para = list( set([ each.split('_')[2*itr_para+5] for each in parse_str ]) )
		ls_para.sort()
		dict_para = dict( zip( [name_para[itr_para]+'_' + each for each in ls_para],  range(0, len(ls_para) ))  )
		dict_para_tol.update({name_para[itr_para]:dict_para})
	
	
	# Tuples for number of parameters 
	n_para_tuples = tuple([ len(dict_para_tol[each]) for each in ['K', 'sim']+name_para]) + (all_rounds, )
	
	tol_reward = np.zeros( n_para_tuples )
	tol_regret = np.zeros( n_para_tuples )
		
	for filename in all_filenames:
		
		results = np.load( './' + save_str + '/' + method + '/' + filename )
		
		regret = results['regret']
		reward = results['reward']
		
		para_str = filename.replace('.npz','').split('_')
		
		# Get parameter strings
		index = [ dict_para_tol[para_str[each]][para_str[each] + '_' + para_str[each+1]] for each in range( 0, 2*(n_para+2), 2) ]
		
		tol_reward[tuple(index)] = reward
		tol_regret[tuple(index)] = regret
		
		#tol_reward[tuple(index), :] = reward.sum()
		#tol_regret[tuple(index), :] = regret.sum()
	
	tol_reward_sum = tol_reward.sum(-1)
	tol_regret_sum = tol_regret.sum(-1)
	
	tol_reward_mean = tol_reward_sum.mean(1)
	tol_regret_mean = tol_regret_sum.mean(1)
	
	# Get the parameters
	ls_best_reward_para = [];
	ls_best_regret_para = [];
	
	for each_K in range(0,  len(ls_K) ):
		
		max_reward_para   = np.unravel_index( np.argmax( tol_reward_mean[each_K].flatten(order='C') ), n_para_tuples[2:-1],order='C' )
		min_regret_para   = np.unravel_index( np.argmin( tol_regret_mean[each_K].flatten(order='C') ), n_para_tuples[2:-1],order='C' )
		
		ls_reward = []
		ls_regret = []
		for itr in range(0, n_para):
			index = list( dict_para_tol[name_para[itr]].values() ).index( max_reward_para[itr] )
			paras = list( dict_para_tol[name_para[itr]].keys() )[index]
			ls_reward.append(paras) 
			
			index = list( dict_para_tol[name_para[itr]].values() ).index( min_regret_para[itr] )
			paras = list( dict_para_tol[name_para[itr]].keys() )[index]
			ls_regret.append(paras)
			
		ls_reward = '_'.join( ls_reward )
		ls_regret = '_'.join( ls_regret )
		
		ls_best_reward_para.append('K_'+ ls_K[each_K] +'_'+ ls_reward)
		ls_best_regret_para.append('K_'+ ls_K[each_K] +'_'+ ls_regret)
			
	if (sim_phase == "val_phase_cal"):
		# Save the best parameter
		np.savez('./val_para/'+method+'.npz', tol_reward=tol_reward, tol_regret=tol_regret, dict_para_tol=dict_para_tol, ls_best_reward_para=ls_best_reward_para, ls_best_regret_para=ls_best_regret_para)
	
	elif(sim_phase == "te_phase_cal"):
		val_para = np.load('./val_para/'+method+'.npz')
		
		tol_reward_test = []
		tol_regret_test = []
		
		for itr_K, para_str in zip( range(0, len(ls_K)), val_para['ls_best_reward_para']):
			
			tol_reward_sel = tol_reward[itr_K,:]
			tol_regret_sel = tol_regret[itr_K,:]
			
			all_index = []
			for itr in range(0, n_para):
				index = dict_para_tol[ para_str.split('_')[2*itr+2] ][para_str.split('_')[2*itr+2]+'_'+para_str.split('_')[2*itr+3]]
				all_index.append(index)
				
				tol_reward_sel = tol_reward_sel[:, index]
				tol_regret_sel = tol_regret_sel[:, index]
			
			tol_reward_test.append(tol_reward_sel)
			tol_regret_test.append(tol_regret_sel)
		
		tol_reward=np.array(tol_reward_test)
		tol_regret=np.array(tol_regret_test)
		
		print("Total reward on the test set for method "+method+" : ",  [tol_reward[each_K].cumsum(-1)[:, -1].mean(0) for each_K in range(0, len(ls_K))] )
		np.savez('./test_results/'+method+'.npz', tol_reward=np.array(tol_reward_test), tol_regret=np.array(tol_regret_test) )

