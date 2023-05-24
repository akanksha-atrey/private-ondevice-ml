'''
Author: Akanksha Atrey
Description: This file contains implementation of an autoencoder based defense mechanism.
'''

from src.defense.defense_training import Autoencoder, Net
import os
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import time
import random
from hwcounter import count, count_end

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, pairwise_distances
from scipy.stats import entropy
from scipy import spatial
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('text', usetex=False)
plt.rcParams['axes.labelsize']=16

DEVICE = None

def detector(model, ae_model, x, prior_queries_raw, prior_queries_enc, prior_stats, prior_outputs, data_type='train', user_id=-1, model_type='rf'):
	out_decoder, out_encoder = ae_model(torch.from_numpy(x.values).float())
	encoding = out_encoder.detach().numpy()

	if model_type == 'dnn':
		pred = np.argmax(model(out_encoder.detach().reshape(1,-1)).data.numpy())
	else:
		pred = model.predict(encoding.reshape(1,-1))[0]

	stime_ed = time.time()
	start_cycles_ed = count()
	ed_raw_min, ed_raw_median, ed_raw_sum = 0, 0, 0
	ed_enc_min, ed_enc_median, ed_enc_sum = 0, 0, 0
	# Difference in input to closest element in class c
	if prior_queries_raw is not None:
		prior_queries_rc = prior_queries_raw[prior_queries_raw['pred_c']==pred]
		prior_queries_ec = prior_queries_enc[prior_queries_enc['pred_c']==pred]

		if len(prior_queries_rc) != 0:
			distances_raw = spatial.distance.cdist(np.array(x).reshape(1,-1), prior_queries_rc.drop(columns=['pred_c', 'user_id']), metric='euclidean')
			ed_raw_min = distances_raw.min(axis=1)[0]
			ed_raw_median = np.median(distances_raw, axis=1)[0]
			ed_raw_sum = np.sum(distances_raw)

			distances_ec = spatial.distance.cdist(encoding.reshape(1,-1), prior_queries_ec.drop(columns=['pred_c', 'user_id']), metric='euclidean')
			ed_enc_min = distances_ec.min(axis=1)[0]
			ed_enc_median = np.median(distances_ec, axis=1)[0]
			ed_enc_sum = np.sum(distances_ec)
	ed_time = time.time() - stime_ed
	ed_cycles = count_end() - start_cycles_ed

	# Reconstruction error (x vs ae(x)) to separate out anomalies/OOD queries
	stime_mse = time.time()
	start_cycles_mse = count()
	mse = (x - out_decoder.detach().numpy())**2
	q75_mse, q25_mse = np.percentile(mse, [75 ,25])
	mse_time = time.time() - stime_mse
	mse_cycles = count_end() - start_cycles_mse

	# Construct entropy of classes
	stime_entropy = time.time()
	start_cycles_entropy = count()
	prior_outputs = dict.fromkeys(range(6),0) if prior_outputs is None else prior_outputs
	prior_outputs[pred] += 1
	output_probability = [v/sum(prior_outputs.values()) for k,v in prior_outputs.items()]
	output_entropy = entropy(output_probability)
	entropy_time = time.time() - stime_entropy
	entropy_cycles = count_end() - start_cycles_entropy

	# Save output
	x['pred_c'] = pred
	x['user_id'] = user_id
	encoding_df = pd.Series(encoding)
	encoding_df['pred_c'] = pred
	encoding_df['user_id'] = user_id
	stats = {'user_id': user_id, 'class': pred, 'data': data_type, \
			'ed_raw_min': ed_raw_min, 'ed_raw_median': ed_raw_median, 'ed_raw_sum': ed_raw_sum, \
			'ed_enc_min': ed_enc_min, 'ed_enc_median': ed_enc_median, 'ed_enc_sum': ed_enc_sum, \
			'mse_mean': np.mean(mse), 'mse_iqr': q75_mse-q25_mse, \
			'output_entropy': output_entropy, \
			'ed_time': ed_time, 'ed_cycles': ed_cycles, \
			'mse_time': mse_time, 'mse_cycles': mse_cycles, \
			'entropy_time': entropy_time, 'entropy_cycles': entropy_cycles}
	
	prior_queries_raw = pd.DataFrame(x).T if prior_queries_raw is None else prior_queries_raw.append(x, ignore_index=True)
	prior_queries_enc = pd.DataFrame(encoding_df).T if prior_queries_enc is None else prior_queries_enc.append(encoding_df, ignore_index=True)

	prior_stats = pd.DataFrame(stats, index=[0]) if prior_stats is None else prior_stats.append(stats, ignore_index=True)
	prior_stats['mse_cumulative_median'] = prior_stats.groupby(['class', 'user_id'])['mse_mean'].apply(lambda x: x.shift().expanding().median())
	prior_stats['mse_cumulative_sum'] = prior_stats.groupby(['class', 'user_id'])['mse_mean'].apply(lambda x: x.shift().expanding().sum())
	prior_stats['ed_enc_cumulative_sum'] = prior_stats.groupby(['class', 'user_id'])['ed_enc_median'].apply(lambda x: x.shift().expanding().sum())

	return prior_queries_raw, prior_queries_enc, prior_stats, prior_outputs

def analyze(model, ae_model, X, indices, model_type='rf'):
	X_subset = X.loc[indices]
	out_decoder, out_encoder = ae_model(torch.from_numpy(X_subset.values).float())
	if model_type == 'dnn':
		_, pred = torch.max(model(out_encoder).data, 1)
		pred = pred.numpy()
	else:
		pred = model.predict(out_encoder.detach())

	out = []
	for c in pred:
		c_idx = np.where(pred==c)[0]
		X_subset_c = X_subset.iloc[c_idx]
		out_encoder_c = out_encoder.detach().numpy()[c_idx]
		out_decoder_c = out_decoder.detach().numpy()[c_idx]

		# Difference in input
		cs_raw = np.mean(pairwise_distances(X_subset_c.values, metric="cosine"), axis=1)
		q75_cs, q25_cs = np.percentile(cs_raw, [75 ,25])
		cs_enc = np.mean(pairwise_distances(out_encoder_c, metric="cosine"), axis=1)
		q75_cs_enc, q25_cs_enc = np.percentile(cs_enc, [75 ,25])
		cs_dec = np.mean(pairwise_distances(out_decoder_c, metric="cosine"), axis=1)

		# Reconstruction error (x vs ae(x)) to separate out anomalies/OOD queries
		mse_train = ((X_subset_c - out_decoder_c)**2).mean(axis=1)
		q75_mse, q25_mse = np.percentile(mse_train, [75 ,25])

		out.append({'class': c,
		  	'num_samples': len(X_subset),
			'cs_median_raw': np.median(cs_raw), 
			'cs_var_raw': np.var(cs_raw), 
			'cs_iqr_raw': q75_cs-q25_cs,
			'cs_median_enc': np.median(cs_enc), 
			'cs_var_enc': np.var(cs_enc),
			'cs_iqr_enc': q75_cs_enc-q25_cs_enc,
			'cs_median_dec': np.median(cs_dec), 
			'cs_var_dec': np.var(cs_dec),
			'mse_median': np.mean(mse_train), 
			'mse_var': np.var(mse_train), 
			'mse_iqr': q75_mse-q25_mse})

	return pd.DataFrame(out)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-data_name', type=str, default='UCI_HAR')
	parser.add_argument('-model_type', type=str, default='rf', help='rf, lr or dnn')
	parser.add_argument('-num_users', type=int, default=50)
	parser.add_argument('-num_queries', type=int, default=100)
	parser.add_argument('-noise_bounds', type=float, default='-1.0 1.0', nargs=2, help='noise will be drawn from uniform distribution between bounds')
	args = parser.parse_args()

	# Load dataset
	X_train = pd.read_csv('./data/UCI_HAR/train/X_train.txt', delim_whitespace=True, header=None)
	y_train = pd.read_csv('./data/UCI_HAR/train/y_train.txt', delim_whitespace=True, header=None).squeeze()
	X_test = pd.read_csv('./data/UCI_HAR/test/X_test.txt', delim_whitespace=True, header=None)
	y_test = pd.read_csv('./data/UCI_HAR/test/y_test.txt', delim_whitespace=True, header=None).squeeze()

	y_train = y_train-1
	y_test = y_test-1

	print("Train dataset shapes: {}, {}".format(X_train.shape, y_train.shape))
	print("Test dataset shapes: {}, {}".format(X_test.shape, y_test.shape))

	# Load subject information for train and test sets
	subject_train = pd.read_csv('./data/UCI_HAR/train/subject_train.txt', delim_whitespace=True, header=None)
	subject_test = pd.read_csv('./data/UCI_HAR/test/subject_test.txt', delim_whitespace=True, header=None)

	print("Number of users in train set: ", subject_train.nunique()[0])
	print("Number of users in test set: ", subject_test.nunique()[0])

	# Load models
	ae_model = torch.load('./models/{}/attack_defense/autoencoder.pt'.format(args.data_name))

	if args.model_type == 'rf':
		with open('./models/UCI_HAR/attack_defense/rf.pkl', 'rb') as f:
			model = pkl.load(f)
	elif args.model_type == 'lr':
		with open('./models/UCI_HAR/attack_defense/lr.pkl', 'rb') as f:
			model = pkl.load(f)
	elif args.model_type == 'dnn':
		model = torch.load('./models/UCI_HAR/attack_defense/dnn.pt')

	# Detect sample wise
	random.seed(8)
	np.random.seed(8)
	rand_nums = random.sample(range(len(X_test)), args.num_users)
	X_rand_users = X_test.iloc[rand_nums]
	y_rand_users = y_test.iloc[rand_nums]

	simulated_queries = pd.concat([X_rand_users]*args.num_queries)
	simulated_queries[simulated_queries.columns] = np.random.uniform(-1,1,size=len(simulated_queries)*simulated_queries.shape[1]).reshape(len(simulated_queries),-1)
	simulated_queries = simulated_queries

	noise_queries = pd.concat([X_rand_users]*args.num_queries)
	rand_noise = np.random.uniform(args.noise_bounds[0],args.noise_bounds[1],size=noise_queries.shape[0]*noise_queries.shape[1]).reshape(noise_queries.shape[0],-1)
	noise_queries = noise_queries + rand_noise
	noise_queries = noise_queries.clip(-1,1)

	# train data
	print('Running detector on train data now...')
	prior_stats = None
	for i in subject_train[0].unique():
		prior_queries_raw, prior_queries_enc, prior_outputs = None, None, None
		user_indices = subject_train[subject_train[0]==i].index
		X_subset = X_train.loc[user_indices]
		for j,row in X_subset.iterrows():
			prior_queries_raw, prior_queries_enc, prior_stats, prior_outputs = detector(model, ae_model, row, prior_queries_raw, prior_queries_enc, prior_stats, prior_outputs, data_type='train', user_id=i, model_type=args.model_type)
	prior_stats.to_csv('./results/UCI_HAR/defense/detector_train_{}.csv'.format(args.model_type), index=False)

	# test data
	print('Running detector on test data now...')
	prior_stats = None
	for i in subject_test[0].unique():
		prior_queries_raw, prior_queries_enc, prior_outputs = None, None, None
		user_indices = subject_test[subject_test[0]==i].index
		X_subset = X_test.loc[user_indices]
		for j,row in X_subset.iterrows():
			prior_queries_raw, prior_queries_enc, prior_stats, prior_outputs = detector(model, ae_model, row, prior_queries_raw, prior_queries_enc, prior_stats, prior_outputs, data_type='test', user_id=i, model_type=args.model_type)
	prior_stats.to_csv('./results/UCI_HAR/defense/detector_test_{}.csv'.format(args.model_type), index=False)

	# random queries
	print('Running detector on random data now...')
	prior_stats = None
	for i in simulated_queries.index.unique():
		prior_queries_raw, prior_queries_enc, prior_outputs = None, None, None
		X_subset = simulated_queries.loc[i]
		for j,row in X_subset.iterrows():
			prior_queries_raw, prior_queries_enc, prior_stats, prior_outputs = detector(model, ae_model, row, prior_queries_raw, prior_queries_enc, prior_stats, prior_outputs, data_type='rand', user_id=i, model_type=args.model_type)
	prior_stats.to_csv('./results/UCI_HAR/defense/detector_rand_{}.csv'.format(args.model_type), index=False)

	# noise queries
	print('Running detector on noise-added data now...')
	prior_stats = None
	for i in noise_queries.index.unique():
		prior_queries_raw, prior_queries_enc, prior_outputs = None, None, None
		X_subset = noise_queries.loc[i]
		for j,row in X_subset.iterrows():
			prior_queries_raw, prior_queries_enc, prior_stats, prior_outputs = detector(model, ae_model, row, prior_queries_raw, prior_queries_enc, prior_stats, prior_outputs, data_type='noise', user_id=i, model_type=args.model_type)
	prior_stats.to_csv('./results/UCI_HAR/defense/detector_noise_{}.csv'.format(args.model_type), index=False)

if __name__ == '__main__':
	main()