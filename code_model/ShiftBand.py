from multiprocessing import Process, Queue
import queue
import shelve
import pickle
import os
import time
import numpy as np
import pdb
import math

def ShiftBand(job_sim, method, sim_phase):
	
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
			itr_K, itr_sim, gamma, alpha, beta, eta	= job_sim.get_nowait()
			itr_K = int(itr_K)
			
			np.random.seed(itr_sim)
			
			if(sim_phase == "val_phase"):
				sim_path_out = './' + save_dir + '/' + method + '/'+ 'K_'+ "{:02d}".format(int(itr_K)) + '_sim_'  + "{:04d}".format(int(itr_sim)) +\
							   "_gamma_" + "{:04f}".format(float(gamma)) + "_alpha_" + "{:04f}".format(float(alpha)) + \
							   "_beta_" + "{:04f}".format(float(beta)) + "_eta_" + "{:04f}".format(float(eta)) + '.npz'
			elif(sim_phase == "te_phase"):
				sim_path_out = './' + save_dir + '/' + method + '/'+ 'K_'+ "{:02d}".format(int(itr_K)) + '_sim_' + "{:04d}".format(int(itr_sim)) +\
							   "_gamma_" + "{:04f}".format(float(gamma)) + "_alpha_" + "{:04f}".format(float(alpha)) + \
							   "_beta_" + "{:04f}".format(float(beta)) + "_eta_" + "{:04f}".format(float(eta)) + '.npz'

			if os.path.exists(sim_path_out):
				 continue

			#if os.path.exists(sim_path_out):
			#	 continue
				
			reward = [];
			regret = [];
			
			N_hist = [None]*Ktol;
			rwd_hist = [None]*Ktol;
			
			n_k = np.zeros((Ktol, ))
			w_k = np.ones((Ktol, ))
			
			for rounds in range(0, all_rounds):
				
				# Set probability 
				probs = gamma/Ktol + (1-gamma) * w_k/w_k.sum()
				
				a_sel = np.random.choice(Ktol, itr_K, p=probs)

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
				
				x_hat = np.zeros((Ktol, ))
				x_hat[a_sel] = Rwd/probs[a_sel]
				
				exp_x_hat = np.exp( eta * ( x_hat+ alpha/(probs*np.sqrt(  Ktol*all_rounds/all_rounds   )) ) )
				#print( "exp_x_hat: ", exp_x_hat, "eta: ", eta )
				#if exp_x_hat[0] > 10**2:
					#print("::exp_x_hat ", w_k, exp_x_hat)
					#pdb.set_trace()
				
				#if np.isinf(  np.multiply( exp_x_hat, w_k ) + beta/Ktol*w_k.sum() ).any():
					#print(probs)
					#pdb.set_trace()
				
				#if np.multiply( exp_x_hat, w_k )[0] > 10**2:
					#print("Not yet: ",  w_k)
					#pdb.set_trace()
				
				w_k_temp = np.copy(w_k)
				w_k = np.multiply( exp_x_hat, w_k ) + beta/Ktol*w_k.sum()
				
				#if w_k[0] != 1:
				#print("exp_x_hat: ", exp_x_hat, w_k, probs, x_hat, eta* ( x_hat+ alpha/(probs*np.sqrt(  Ktol*int(Tmax/deltaT)/int(Tmax/deltaT)   )) ))
					#pdb.set_trace()
			
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
