from multiprocessing import Process, Queue
import queue
import shelve
import pickle
import os
import time
import numpy as np
import pdb
import math

def m_ucb(job_sim, method, sim_phase):
	
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

	# Modif = y the results
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
			itr_K, itr_sim, window, thresh, gamma = job_sim.get(False)
			itr_K = int(itr_K)
			
			np.random.seed(itr_sim)
			
			if(sim_phase == "val_phase"):
				sim_path_out = './' + save_dir + '/' + method + '/'+ 'K_'+ "{:02d}".format(int(itr_K)) + '_sim_'  + "{:04d}".format(int(itr_sim)) +\
							   "_window_" + "{:04d}".format(int(window)) + "_thresh_" + "{:04f}".format(float(thresh)) + "_gamma_" + "{:04f}".format(float(gamma)) +'.npz'
			elif(sim_phase == "te_phase"):
				sim_path_out = './' + save_dir + '/' + method + '/'+ 'K_'+ "{:02d}".format(int(itr_K)) + '_sim_' + "{:04d}".format(int(itr_sim)) +\
							   "_window_" + "{:04d}".format(int(window)) + "_thresh_" + "{:04f}".format(float(thresh)) + "_gamma_" + "{:04f}".format(float(gamma)) + '.npz'

			if os.path.exists(sim_path_out):
				 continue

			#if os.path.exists(sim_path_out):
			#	 continue
				
			reward = [];
			regret = [];
			
			
			N_hist = [None]*Ktol;
			
			n_k = np.zeros((Ktol, ))
			U_k = np.ones((Ktol, )) * np.inf
			
			tao = 0
			for rounds in range(0, all_rounds):
				
				a_Psel = np.int( np.mod( (rounds - tao), np.floor(Ktol/gamma) ) )
				
				if a_Psel <= Ktol:
					ls_inf = np.where( n_k==0 )[0]
					if ls_inf.shape[0] > itr_K:
						a_sel = np.random.choice( ls_inf, itr_K)
					else:
						get = np.random.choice( np.setdiff1d( np.arange(0, Ktol), ls_inf), itr_K-len(ls_inf) )
						get = np.hstack((ls_inf, get))
					
				else:
					# Update All UCB				
					for each_K in range(0, Ktol):
						try:
							U_k[each_K] = np.mean( N_hist[each_K] ) + np.sqrt( 2*np.log(rounds-tao)/n_k[each_K]   )
						except:
							U_k[each_K] = np.inf
					
					
					a_sel = np.lexsort( (np.random.random(U_k.size), U_k) )[::-1][0:itr_K]
					#a_sel = np.argmax( U_k )
				
				Rwd = N_Ts[:, rounds]
				Rwd = Rwd[a_sel].sum()

				n_k[a_sel] = n_k[a_sel] + 1

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
					else:
						N_hist[each_arm].append(Rwd)
				
				for each_arm in a_sel:
					# CD algo	
					if n_k[each_arm] >= window:
					
						if np.abs(np.sum( N_hist[each_arm][ 0: np.int(np.round(window/2)) ] ) - np.sum( N_hist[each_arm][ np.int(np.round(window/2)): ] )) > thresh:
							n_k = np.zeros((Ktol, ))
							tao = rounds
				
				
				str_each_UCB = [ "{:9.4f}".format(U_k[itr]) for itr in range(0, Ktol)]
				str_each_UCB = " ".join(str_each_UCB)
				
				#print(reward[-1])
				
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
		#except queue.Empty:
		except Exception as e:
			#print(e)
			if job_sim.qsize()==0:
				#print("Empty: Break_out", job_sim.qsize(), queue.Empty)
				break
		else:
			#print("else: ")
			time.sleep(.5)
	
	return 0
