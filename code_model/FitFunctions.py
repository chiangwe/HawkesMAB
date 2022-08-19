import math 
import numpy as np

class FitFunctions:
	
	#x0 = [mu, alpha, beta];
	def __init__(self, rounds, deltaT):

		self.rounds = rounds
		self.deltaT = deltaT
		
		pass

	def Exp_L(self, x0, timestamp):
		
		mu, alpha, beta = x0[0], x0[1], x0[2]
		if beta==alpha:
			#timestamp = (self.rounds+1)*self.deltaT
			exp_N = mu	+ mu*alpha*timestamp
		else:
			#timestamp = (self.rounds+1)*self.deltaT
			exp_N = -beta*mu/(alpha-beta) \
					+ alpha*mu/((alpha-beta)**1) \
						* math.exp((alpha-beta)*timestamp)
		
		return exp_N
	
	def Exp_N(self, x0, timestamp):
		
		mu, alpha, beta = x0[0], x0[1], x0[2]
		
		#pdb.set_trace()
		if beta==alpha:
			#timestamp = (self.rounds+1)*self.deltaT
			exp_N = mu * timestamp + \
					1/2 * mu*alpha*(timestamp**2)
		else:
			#timestamp = (self.rounds+1)*self.deltaT
			exp_N = -beta*mu*timestamp/(alpha-beta) \
					+ alpha*mu/((alpha-beta)**2) \
						* (math.exp((alpha-beta)*timestamp)-1)
		
		return exp_N

	def Exp_N_Lambda(self, x0, timestamp):
		
		mu, alpha, beta = x0[0], x0[1], x0[2]
		
		self.D21 = alpha*beta*mu/((alpha-beta)**2) + \
			  (  (beta**2)*(mu**2) + (alpha*beta*(mu**2)) \
				  - 0.5*beta*mu*((alpha**2)+2*beta*mu)	  )/((alpha-beta)**3)
		
		self.D22 = ((beta**2)*(mu**2))/((alpha-beta)**2)
		self.D23 = (-(mu**2))/(alpha-beta) - (alpha*beta*mu)/((alpha-beta)**2) - \
			  ( ((beta**2)*(mu**2)) + (alpha*beta*mu) - 0.5*beta*mu*(alpha**2+2*beta) +\
				 mu*(alpha**2+2*beta*mu)*(alpha-0.5*beta) ) / ( (alpha-beta)**3 )
		
		self.D24 = (alpha**2)*mu/(alpha-beta) + \
			  ((alpha*beta*(mu**2)) - alpha*mu*((alpha**2)+2*beta*mu))/((alpha-beta)**2)
		
		self.D25 = ((mu**2)/(alpha-beta)) + (mu*((alpha**2)+2*beta*mu)*(alpha-0.5*beta))/((alpha-beta)**3)
		
		self.D31 = alpha * mu + (mu**2)
		self.D32 = 0.5*beta*(mu**2) + 0.5*(alpha**2)*mu - 0.5*mu*(alpha**2 + 2*beta*mu)
		self.D33 = 1/6*alpha*beta*(mu**2) + 1/6*alpha*mu*(alpha**2 + 2*beta*mu)
		
		if beta==alpha:
			#timestamp = (self.rounds+1)*self.deltaT
			exp_N_Lambda = self.D31*timestamp + self.D32*(timestamp**2) + self.D33*(timestamp**3)
		else:
			#timestamp = (self.rounds+1)*self.deltaT
			exp_N_Lambda = self.D21 + self.D22*timestamp + self.D23*np.exp((alpha-beta)*timestamp) +\
						   self.D24*timestamp*np.exp((alpha-beta)*timestamp) +\
						   self.D25*np.exp(2*(alpha-beta)*timestamp)
			
		return exp_N_Lambda
	
	def Exp_N_sqare(self, x0, timestamp):
		
		mu, alpha, beta = x0[0], x0[1], x0[2]
		
		self.Exp_N_Lambda(x0, timestamp)
		
		self.D42 = -beta*mu/(alpha-beta) + 2*self.D21
		self.D43 = self.D22
		self.D44 = alpha*mu/((alpha-beta)**2) + 2*self.D23/(alpha-beta)\
				   -2*self.D24/((alpha-beta)**2)
		self.D45 =	 self.D25/(alpha-beta)
		self.D46 = 2*self.D24/(alpha-beta)
		self.D41 = -self.D44-self.D45
		
		self.D72 = mu
		self.D73 = 0.5*mu*alpha + self.D31
		self.D74 = 2/3 * self.D32
		self.D75 = 0.5 * self.D33
		
		if beta==alpha:
			#timestamp = (self.rounds+1)*self.deltaT
			exp_N_square =	self.D72*timestamp + self.D73*(timestamp**2) +\
							self.D74*(timestamp**3) + self.D75*(timestamp**4)
		else:
			#timestamp = (self.rounds+1)*self.deltaT
			exp_N_square = self.D41 + self.D42*timestamp + self.D43*(timestamp**2) + \
						   self.D44*np.exp((alpha-beta)*timestamp) + \
						   self.D45*np.exp(2*(alpha-beta)*timestamp) +\
						   self.D46*timestamp*np.exp((alpha-beta)*timestamp)
			
		return exp_N_square

	def Var_N(self, x0, timestamp):
		
		mu, alpha, beta = x0[0], x0[1], x0[2]
		
		self.Exp_N_sqare(x0, timestamp)
		
		self.D51 = self.D41 - ((alpha**2)*(mu**2))/((alpha-beta)**4)
		self.D52 = self.D42 - (2*alpha*beta*(mu**2))/((alpha-beta)**3)
		self.D53 = self.D43 - ((beta**2)*(mu**2))/((alpha-beta)**2)
		self.D54 = self.D44 + (2*(alpha**2)*(mu**2))/((alpha-beta)**4)
		self.D55 = self.D45 - ((alpha**2)*(mu**2))/((alpha-beta)**4)
		self.D56 = self.D46 + (2*alpha*beta*(mu**2))/((alpha-beta)**3)
		
		if beta==alpha:
			#timestamp = (self.rounds+1)*self.deltaT
			var_N = np.nan
		else:
			#timestamp = (self.rounds+1)*self.deltaT
			#pdb.set_trace()
			var_N = self.D51 + self.D52*timestamp + self.D53*(timestamp**2) + \
						   self.D54*np.exp((alpha-beta)*timestamp) + \
						   self.D55*np.exp(2*(alpha-beta)*timestamp) +\
						   self.D56*timestamp*np.exp((alpha-beta)*timestamp)
			
		return var_N
	
	def Exp_Phi(self, x0):
		
		mu, alpha, beta = x0[0], x0[1], x0[2]
		
		if beta==alpha:
			exp_Phi = mu * self.deltaT + \
					  1/2 *mu*alpha*self.deltaT*(2*self.rounds+self.deltaT)
		else:
			exp_Phi = -beta*mu*self.deltaT/(alpha-beta) + \
					   alpha*mu/((alpha-beta)**2) * \
					   math.exp((alpha-beta)*self.deltaT*self.rounds) * \
					   (math.exp((alpha-beta)*self.deltaT)-1)			
		return exp_Phi

	def Var_Phi(self, x0):
		
		mu, alpha, beta = x0[0], x0[1], x0[2]
		if beta==alpha:
			timestamp = (self.rounds+1)*self.deltaT
			var_Phi = np.nan
		else:
			
			timestamp_prev = (self.rounds  )*self.deltaT
			timestamp_now  = (self.rounds+1)*self.deltaT
			
			# Var_N prev
			exp_n_prev	  = self.Exp_N(		   x0, timestamp_prev )
			exp_n_lm_prev = self.Exp_N_Lambda( x0, timestamp_prev )
			exp_n_sq_prev = self.Exp_N_sqare(  x0, timestamp_prev )
			var_n_prev	  = self.Var_N(		   x0, timestamp_prev )
			
			# Var_N
			exp_n_now	 = self.Exp_N(		  x0, timestamp_now )
			exp_n_lm_now = self.Exp_N_Lambda( x0, timestamp_now )
			exp_n_sq_now = self.Exp_N_sqare(  x0, timestamp_now )
			var_n_now	 = self.Var_N(		  x0, timestamp_now )
			
			#pdb.set_trace()
			var_Phi = var_n_now + var_n_prev + 2*exp_n_prev*exp_n_now+\
					  -2*exp_n_sq_prev + 2*beta*mu*exp_n_prev*(self.deltaT/(alpha-beta))\
					  -2*(np.exp((alpha-beta)*self.deltaT)-1)*\
						 ( beta*mu/((alpha-beta)**2)*exp_n_prev+\
						  exp_n_lm_prev/(alpha-beta) )
			#pdb.set_trace()
			
		return var_Phi
	
	def Exp_Phi_MuTs(self, x0, muTs, rounds):
		
		mu, alpha, beta = x0[0], x0[1], x0[2]
		if beta==alpha:
			pass
		else:
			exp_Phi = -beta*mu*self.deltaT/(alpha-beta) + \
						 ( alpha*mu/((alpha-beta)**2)+ (alpha/(alpha-beta))* \
						   np.sum(np.exp(-beta*muTs)) ) * \
					   math.exp((alpha-beta) * self.deltaT * rounds) * \
					   (math.exp((alpha-beta)*self.deltaT)-1)			
		return exp_Phi
