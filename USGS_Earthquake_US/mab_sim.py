import sys
sys.path.append('../')
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Process, Queue
import queue
from hyper_para import hyper_para
import numpy as np

#from hp_mab import hp_mab
#from ucb1_mab import ucb1_mab
#from hp_mab_MuTs import hp_mab_MuTs
#from m_ucb import m_ucb
#from exp3 import exp3
#from exp3S import exp3S
#from ShiftBand import ShiftBand
#from d_ucb import d_ucb
from code_model.slide_ucb import slide_ucb
from code_model.hp_mab import hp_mab
from code_model.hp_mab_MuTs import hp_mab_MuTs

import itertools

import pdb
import sys
import os

if __name__ == '__main__':
	
	n_sim = int( sys.argv[1] )
	method = sys.argv[2]
	sim_phase = sys.argv[3]
	
	job_sim = Queue()
	
	# Clear results 
	if (sim_phase == "clear"):
		
		# Clear val results
		path_dir = "./val_results/" + method + "/"
		for filename in os.listdir( path_dir ):
			os.remove(path_dir + filename)
		
		# Clear val para
		path_dir = "./val_para/" + method + ".npz"
		if os.path.exists(path_dir):
			os.remove( path_dir )
		
		# Clear test results
		path_dir = "./results/" + method + "/"
		for filename in os.listdir( path_dir ):
			os.remove( path_dir + filename )
		
		# Clear results
		path_dir = "./test_results/" + method + "/"
		if os.path.exists(path_dir):
			os.remove( path_dir )
	
	# Finding the hyper parameter phase: val_phase
	list_K = [1, 3, 5]
	if (sim_phase == "val_phase"):
		
		if method == 'slide_ucb':
			
			list_sim = list( range(0, n_sim) )
			list_tao = [ 20, 50, 100, 300, 500 ];
			
			tol_tuple = list(itertools.product( list_K, list_sim, list_tao))		
			
		elif (method == 'hp_mab') | (method == 'hp_mab_MuTs'):
			
			# Sim
			list_sim = list( range(0, n_sim) )
			
			# Epslon Bias
			list_epslon = [50, 100, 300, 500];
			# Eta Scale
			list_eta = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1, 5, 10];
			
			#tol_tuple = [ (each,) for each in list_sim ]
			tol_tuple = list(itertools.product( list_K, list_sim, list_epslon, list_eta))
			
	elif (sim_phase == "te_phase") :
		
		# Get best hyper parameter for reward
		val_para = np.load('./val_para/'+method+'.npz')
		if val_para['ls_best_reward_para'].shape[0] > 0:
			
			#para_str = '_'.join( val_para['ls_best_reward_para'] )
			tol_tuple = []
			list_sim = list( range(0, n_sim) )
			for para_str in val_para['ls_best_reward_para']:

				ls_tuple = tuple( [ float(each) for each in para_str.split('_')[1::2] ] )
				ls_tuple = [ (int(ls_tuple[0]), ) +  (each, ) + ls_tuple[1:]  for each in list( range(0, n_sim) ) ]
				tol_tuple = tol_tuple + ls_tuple
				#tol_tuple = [ (each, ) + tol_tuple  for each_k in list( range(0, n_sim) ) ]
			
		else:
			
			list_sim = list( range(0, n_sim) )
			tol_tuple = [ (each, )  for each in list( range(0, n_sim) ) ]
		
	if (sim_phase == "te_phase") | (sim_phase == "val_phase"):
		
		for each_tup in tol_tuple:
			job_sim.put( each_tup )	
		
		#globals()[method](job_sim, method, sim_phase)
		n_procs = 50
		procs = []
		for itr_proc in range(0, n_procs):
			proc = Process(target=globals()[method], args=(job_sim, method, sim_phase))
			procs.append(proc)
			proc.start()
		
		# complete the processes
		for proc in procs:
			proc.join()
	
	if (sim_phase == "val_phase_cal")|(sim_phase == "te_phase_cal"):
		hyper_para( method, sim_phase )
	
	
	#globals()[method](job_sim, method, sim_phase)
	#proc = Process(target=globals()[method], args=(job_sim,))
	#if method == 'm_ucb':
	#	for itr_sim in range(0, n_sim):
	#		job_sim.put( (itr_sim,(10, 20, 0.5) ) )
	#elif method == 'exp3':
	#	for itr_sim in range(0, n_sim):
	#		job_sim.put( (itr_sim,(0.000001) ) )
	#elif method == 'exp3S':
	#	for itr_sim in range(0, n_sim):
	#		job_sim.put( (itr_sim,(0.000001, 0.001) ) )
	#elif method == 'ShiftBand':
	#	for itr_sim in range(0, n_sim):
	#		job_sim.put( (itr_sim, ( 0.5, 0.001, 0.0001, 0.0001 ) ) )	
	#elif method == 'd_ucb':
	#	for itr_sim in range(0, n_sim):
	#		job_sim.put( (itr_sim, ( 0.1, 1 ) ) )	
	#elif method == 'slide_ucb':
	#	for itr_sim in range(0, n_sim):
	#		job_sim.put( (itr_sim, ( 30, ) ) )	
	#else:
	#	for itr_sim in range(0, n_sim):
	#		job_sim.put(itr_sim)
	
