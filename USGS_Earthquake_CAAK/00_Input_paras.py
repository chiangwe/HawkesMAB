# Source: https://www.kaggle.com/wosaku/crime-in-vancouver

import numpy as np
import pdb
import pandas as pd
import math
import itertools
import pickle

key_f = lambda x: x[0]

# Parameters
tol_rounds = 500 
rounds_tr = 50

# Read processed csv 
df = pd.read_csv('../raw_data/Crime_in_Los_Angeles/crime_processed.csv')

# Sort date time
df = df.sort_values(by="datetime_occur").reset_index(drop=True)

# Get reciever code and time
df = df[['Crime Code 1', 'datetime_occur']]

# Rename the columns
df = df.rename(columns={"Crime Code 1":"crime_type", "datetime_occur":"date_time"})

# Drop the same time stamp for the same type
df = df.drop_duplicates().reset_index(drop=True)

# Count the number of events 
df_size = df.groupby('crime_type').size().sort_values(ascending=False)

# Get events more than 20,000
df_size_sel = df_size[ df_size > 20000 ]
pdb.set_trace()
# REMOVE THE FIRST ONE SINCE IT IS TOO LARGE
#df_size_sel = df_size_sel.iloc[1:]

# #################################

code_sel = df_size_sel.index.tolist()

n_codes = len(code_sel)
Ktol = n_codes

# select the dataframe
df_sel = df[df['crime_type'].isin(code_sel)]

# Formate the timestamp
df_sel['date_time'] = pd.to_datetime( df_sel['date_time'] ).dt.round('1s')
df_sel['unix_s'] = df_sel['date_time'].apply(lambda x: x.timestamp()/60 )

# offset the dataset
unix_min = df_sel['unix_s'].min()
df_sel['unix_s'] = df_sel['unix_s']-unix_min

# Calculate the rounds and step size
unix_min = df_sel['unix_s'].min()
unix_max = df_sel['unix_s'].max()

step_size = math.ceil( unix_max / tol_rounds );
edges = np.arange(0, unix_max+step_size+10**-5, step_size)

N_Ts = np.zeros(( n_codes, tol_rounds ))

# Break down into several dataframes
TsSet = []

for each, itr in zip( code_sel, range(0, len(code_sel)) ):
	
	ts = df_sel[ df_sel['crime_type']==each ] 
	ts = ts.sort_values('unix_s')['unix_s'].tolist()

	inds = np.digitize( ts, edges)
	
	all_ts = [];
	all_key = [];

	for key, group in itertools.groupby( zip( inds, ts), key_f):
		
		all_ts.append( [ each[1] for each in list(group)] )
		all_key.append( key-1 )
	
	dict_all_ts = dict(zip(all_key, all_ts))
	
	all_ts       = [ dict_all_ts.get(each, []) for each in range(0, tol_rounds ) ]
	N_Ts[itr, :] = np.array( [ len(each) for each in all_ts ] )
	
	TsSet.append(all_ts)

# Save timestamp
with open("./data/raw_data.pkl", "wb") as fp:
	pickle.dump( TsSet, fp)
	
# Save number of events
np.save('./data/N_Ts.npy', N_Ts)

# Save parameters
paras = { 'tol_rounds':tol_rounds, 'rounds_tr':rounds_tr, 'step_size':step_size, 'Ktol':Ktol }
with open("./data/paras.pkl", "wb") as fp:
	pickle.dump( paras, fp)

# Save and get the optimal arm
optimal_k = np.argmax(N_Ts, 0)
np.save('./data/optimal_k.npy', optimal_k)

# Rank arm
rank_arm = np.argsort(N_Ts, 0)[::-1,:]
np.save('./data/rank_arm.npy', rank_arm)
