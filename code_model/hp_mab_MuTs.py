from multiprocessing import Process, Queue
import queue
import shelve
import pickle

import itertools
key_f = lambda x: x[0]

import time
import numpy as np
import pdb
import math
from code_model.FitFunctions import FitFunctions
import os 

def hp_mab_MuTs(job_sim, method, sim_phase):
	
	optimal_k = np.load("./data/optimal_k.npy")
	rank_arm = np.load("./data/rank_arm.npy")
	
	# Load Data
	with open("./data/raw_data.pkl", "rb") as fp:
		TsSet = pickle.load( fp)

	# Load Timestamp
	N_Ts = np.load('./data/N_Ts.npy')

	# Load parameter
	with open("./data/paras.pkl", "rb") as fp:
		paras = pickle.load( fp)

	# Load estmated Hawkes Processes
	para_est = np.load("./data/para_est.npz")
	est_all = para_est['arr_para']
	est_bool = para_est['est_bool']

	globals().update( paras )
	
	
	# Modif = y the results
	if(sim_phase == "val_phase"):

		all_rounds = rounds_tr
		N_Ts = N_Ts[:, 0:rounds_tr]
		optimal_k = optimal_k[0:rounds_tr]
		rank_arm = rank_arm[:, 0:rounds_tr]
		
		est_all = est_all[:, :, 0:rounds_tr]
		est_bool = est_bool[:, 0:rounds_tr]

		for each_k in range(0, Ktol):
			TsSet[each_k] = TsSet[each_k][0:rounds_tr]

		save_dir = 'val_results'

	elif(sim_phase == "te_phase"):
		#pdb.set_trace()
		all_rounds = tol_rounds - rounds_tr
		N_Ts = N_Ts[:, rounds_tr:]
		optimal_k = optimal_k[rounds_tr:]
		rank_arm = rank_arm[:, rounds_tr:]
		est_all = est_all[:, :, rounds_tr:]
		est_bool = est_bool[:, rounds_tr:]

		for each_k in range(0, Ktol):
			TsSet[each_k] = [ [each_t - step_size*rounds_tr for each_t in eachTs ]for eachTs in  TsSet[each_k][rounds_tr:] ]

		save_dir = 'results'	
	
	while ( True ):
		try:
			itr_K, itr_sim, epslon, eta = job_sim.get_nowait()
			itr_K = int(itr_K)
			
			np.random.seed(itr_sim)
			
			if(sim_phase == "val_phase"):
				sim_path_out = './' + save_dir + '/' + method + '/'+ 'K_'+ "{:02d}".format(int(itr_K)) + '_sim_'  + "{:04d}".format(int(itr_sim)) +\
							   "_epslon_" + "{:04d}".format(int(epslon)) + "_eta_" + "{:f}".format(float(eta)) + '.npz'
			elif(sim_phase == "te_phase"):
				sim_path_out = './' + save_dir + '/' + method + '/'+ 'K_'+ "{:02d}".format(int(itr_K)) + '_sim_' + "{:04d}".format(int(itr_sim)) +\
							   "_epslon_" + "{:04d}".format(int(epslon)) + "_eta_" + "{:f}".format(float(eta)) + '.npz'

			if os.path.exists(sim_path_out):
				 continue
				
			reward = [];
			regret = [];
			select_arms = np.zeros((itr_K, all_rounds))
			
			para_hist  = [None]*Ktol;
			N_hist	  = [None]*Ktol;
			Phi_hist  = [None]*Ktol;
			
			LastObsTs = [None]*Ktol;
			
			n_k = np.zeros((Ktol, ))
			U_k = np.ones((Ktol, )) * np.inf
			
			for rounds in range(0, all_rounds):
			
				itv_est = [step_size*rounds, step_size*(rounds+1)]
				
				atten = np.ceil( 8*np.log(rounds+1) );
				cand = np.where(n_k<atten)[0]

				cand = np.array([])
				
				if cand.shape[0] >0 :
					
					# if there are some arms left that hasn't meet the exploration requirement
					if cand.shape[0] >= itr_K:
						# More that the number of arms to pull
						a_sel = cand[np.lexsort( (np.random.random(U_k[cand].size), U_k[cand]) )[::-1][0:itr_K]]
					else:
						pdb.set_trace()
						# Candidate has less than the arms to pull
						# Pull cand
						n_left = itr_K - cand.shape[0]
						list_wo_cand = np.setdiff1d(np.arange(0, Ktol), cand)

						a_sel = list_wo_cand[np.lexsort( (np.random.random(list_wo_cand.size), U_k[list_wo_cand]) )[::-1][0:n_left]]
						a_sel = np.hstack((cand, a_sel))
				else:
					a_sel = np.lexsort( (np.random.random(U_k.size), U_k) )[::-1][0:itr_K]	
				
				select_arms[:, rounds] = a_sel
				
				Rwd = N_Ts[:, rounds]
				Rwd = Rwd[a_sel].sum()

				# Update n_k
				n_k[a_sel] = n_k[a_sel] + 1

				#opt_a = np.argmax( phi_the[:, rounds] )
				#opt_a = optimal_k[rounds]
				opt_a = rank_arm[0:itr_K, rounds]

				Rwd_opt = N_Ts[:, rounds]
				Rwd_opt = Rwd_opt[opt_a].sum()

				reward.append( Rwd )
				
				if (Rwd_opt-Rwd) > 0:
					regret.append( Rwd_opt-Rwd )
				else:
					regret.append( 0 )

				rd = rounds+1
				
				for each_arm in a_sel:

					est_para = est_all[:, each_arm, rounds]
					phi = N_Ts[each_arm, rounds]

					if para_hist[each_arm] == None:
						para_hist[each_arm] = [est_para]
					else:
						para_hist[each_arm].append(est_para)

					if N_hist[each_arm] == None:
						N_hist[each_arm] = [phi]
					else:
						N_hist[each_arm].append(phi)
					
					LastObsTs[each_arm] = ( rounds, TsSet[each_arm][rounds] )
					#LastObsTs[each_arm] = ( rounds, [ ] )
				
				
				# Find avg Phit and avg N and bounds
				fitclass = FitFunctions(rd, step_size)
				
				for k in range(0, Ktol):
					#[dict_est['mu'], dict_est['R0'], dict_est['beta'], dict_est['alpha']]
					
					if(para_hist[k] != None):
						list_phi = []
						for each_para in para_hist[k]:
							x0 = np.array( [each_para[0], each_para[3], each_para[2]])
							
							if LastObsTs[k] != None:
								muTs = ( (rounds+1)*step_size) - np.array( LastObsTs[k][1] )
							else:
								muTs = np.array([])
							
							#pdb.set_trace()
							#list_phi.append( fitclass.Exp_Phi(x0) )

							#pdb.set_trace()
							list_phi.append( fitclass.Exp_Phi_MuTs(x0, muTs, rounds - LastObsTs[k][0]) )
							
						#if( rounds > 1100 ):
						#	pdb.set_trace()
						
						mean_phi = np.mean(list_phi)
						#std_phi = np.std(list_phi)
						#std_phi = phi_all_std[k, rounds]
						var_phi = np.square(list_phi).sum() -  (mean_phi**2) * n_k[k] + epslon
						
						if len(list_phi)>=2:
							U_k[k] = mean_phi + \
							np.sqrt( eta*16* var_phi / (n_k[k] -1) * np.log(rounds-1)/n_k[k] )
							#std_phi*100/n_k[k]
							#40*np.sqrt(2*np.log(rounds)/n_k[k])
							#std_phi*0.5/np.sqrt(n_k[k])
							#std_phi*np.sqrt( 2*np.log(std_phi*rounds/np.sqrt(2*np.pi)))/np.sqrt(n_k[k])
							#		  std_phi*np.sqrt( 2*np.log(std_phi*rounds/np.sqrt(2*np.pi)))/np.sqrt(n_k[k])
							# std_phi*np.sqrt( 2*np.log(std_phi*rounds/np.sqrt(2*np.pi)))/n_k[k]
							#pdb.set_trace()
						if np.isnan(U_k[k]):
							U_k[k] = np.inf
		
				
				#print(reward[-1])
				
				#print("sim: ", "{:03d}".format(itr_sim) , \
				#	   "rounds: ", "{:04d}".format(rounds), \
				#	   "a_sel :", "{:02d}".format(a_sel), \
				#	   "UCB: ", "{:9.4f}".format(U_k[a_sel]), \
				#	   "reward: ", "{:06d}".format(int(reward[-1])), \
				#	   "regreT: ", "{:06d}".format(int(regret[-1])), \
				#	   "TolrewardT: ", "{:06d}".format(int(np.sum(reward))), \
				#	   "TolregreT: ", "{:06d}".format(int(np.sum(regret))), U_k )
			
			reward = np.array(reward)
			regret = np.array(regret)
			np.savez( sim_path_out, reward=reward, regret=regret, select_arms=select_arms )
		except Exception as e:
			#print(e)
			if job_sim.qsize()==0:
				#print("Empty: Break_out", job_sim.qsize(), queue.Empty)
				break
		else:
			#print("else: ")
			time.sleep(.5)
	
	return 0
