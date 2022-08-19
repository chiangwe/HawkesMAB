from multiprocessing import Process, Queue
import queue
import shelve
import pickle
import os
import time
import numpy as np
import pdb
import math

def ucb1_mab(job_sim, method, sim_phase):
	
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

	globals().update( paras )

	if(sim_phase == "val_phase"):

		all_rounds = rounds_tr
		N_Ts = N_Ts[:, 0:rounds_tr]
		optimal_k = optimal_k[0:rounds_tr]
		rank_arm = rank_arm[:, 0:rounds_tr]

		for each_k in range(0, Ktol):
			TsSet[each_k] = TsSet[each_k][0:rounds_tr]

		save_dir = 'val_results'

	elif(sim_phase == "te_phase"):

		all_rounds = tol_rounds - rounds_tr
		N_Ts = N_Ts[:, rounds_tr:]
		optimal_k = optimal_k[rounds_tr:]
		rank_arm = rank_arm[:, rounds_tr:]

		for each_k in range(0, Ktol):
			TsSet[each_k] = [ [each_t - step_size*rounds_tr for each_t in eachTs ]for eachTs in  TsSet[each_k][rounds_tr:] ]

		save_dir = 'results'

	while ( True ):
		try:
			itr_K, itr_sim	= job_sim.get_nowait()
			itr_K = int(itr_K)
			
			np.random.seed(itr_sim)
			
			if(sim_phase == "val_phase"):
				sim_path_out = './' + save_dir + '/' + method + '/'+ 'K_'+ "{:02d}".format(int(itr_K)) + '_sim_'  + "{:04d}".format(int(itr_sim)) +\
							   '.npz'
			elif(sim_phase == "te_phase"):
				sim_path_out = './' + save_dir + '/' + method + '/'+ 'K_'+ "{:02d}".format(int(itr_K)) + '_sim_' + "{:04d}".format(int(itr_sim)) +\
							   '.npz'

			if os.path.exists(sim_path_out):
				 continue
				
			reward = [];
			regret = [];
			select_arms = np.zeros((itr_K, all_rounds))
			
			N_hist	  = [None]*Ktol;
			rwd_hist = [None]*Ktol;
			
			n_k = np.zeros((Ktol, ))
			U_k = np.ones((Ktol, )) * np.inf
			
			for rounds in range(0, all_rounds):
			
				itv_est = [step_size*rounds, step_size*(rounds+1)]
				
				a_sel = np.lexsort( (np.random.random(U_k.size), U_k) )[::-1][0:itr_K]
				
				# Update n_k
				Rwd = N_Ts[:, rounds]
				Rwd = Rwd[a_sel].sum()

				n_k[a_sel] = n_k[a_sel] + 1

				#opt_a = optimal_k[rounds]
				opt_a = rank_arm[0:itr_K, rounds]

				Rwd_opt = N_Ts[:, rounds]
				Rwd_opt = Rwd_opt[opt_a].sum()

				reward.append( Rwd )

				if (Rwd_opt-Rwd) > 0:
					regret.append( Rwd_opt-Rwd )
				else:
					regret.append( 0 )

				for each_arm in a_sel:
					if N_hist[each_arm] == None:
						N_hist[each_arm] = [Rwd]
						rwd_hist[each_arm] = [rounds]
					else:
						N_hist[each_arm].append(Rwd)
						rwd_hist[each_arm].append(rounds)
				
				for k in range(0, Ktol):
					if (N_hist[k] == None):
						mean_phi = np.inf
						U_k[k] = np.inf
					else:
						if len( N_hist[k])==0:
							mean_phi = np.inf
							U_k[k] = np.inf
						else:
							mean_phi = np.mean( N_hist[k] )
							U_k[k] = mean_phi + np.sqrt( 2*np.log(rounds+1)  / (len(rwd_hist[k])) )

					if np.isnan(U_k[k]):
						U_k[k] = np.inf
				
				#for k in range(0, Ktol):
				#	
				#	if N_hist[k] != None:
				#		mean_phi = np.mean( N_hist[k] )
				#	else:
				#		mean_phi = np.inf
				#		
				#	U_k[k] = mean_phi + np.sqrt( 2*np.log(rounds+1) / (n_k[k] ) )
				#	
				#	if np.isnan(U_k[k]):
				#		U_k[k] = np.inf
				str_each_UCB = [ "{:9.4f}".format(U_k[itr]) for itr in range(0, Ktol)]
				str_each_UCB = " ".join(str_each_UCB)
				
				
				#print("sim: ", "{:03d}".format(itr_sim) , \
				#	   "rounds: ", "{:04d}".format(rounds), \
				#	   "a_sel :", "{:02d}".format(a_sel), \
				#	   "UCB: ", "{:9.4f}".format(U_k[a_sel]), \
				#	   "reward: ", "{:06d}".format(int(reward[-1])), \
				#	   "regreT: ", "{:06d}".format(int(regret[-1])), \
				#	   "TolregreT: ", "{:06d}".format(int(np.sum(regret))), \
				#	   "\nAll UCB: ", str_each_UCB)
			
			reward = np.array(reward)
			regret = np.array(regret)
			np.savez( sim_path_out, reward=reward, regret=regret )
		except Exception as e:
			#print(e)
			if job_sim.qsize()==0:
				#print("Empty: Break_out", job_sim.qsize(), queue.Empty)
				break
		else:
			#print("else: ")
			time.sleep(.5)
	
	return 0
