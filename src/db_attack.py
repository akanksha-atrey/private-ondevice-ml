'''
Author: Akanksha Atrey
Description: This file contains implementation of attacks on trained ML models to exploit decision boundaries.
'''
from src.models_har import Net

import os
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import random
import torch

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import matplotlib.pyplot as plt

def attack_db_query_distance(model, X_seed_for_classes, model_type='rf', num_queries=1000):
	df_out = pd.DataFrame(columns=['model_type', 'num_query', 'seed_class', 'num_same_class', \
							'cos_sim_same', 'cos_sim_diff', 'cos_sim_same_std', 'cos_sim_diff_std', \
							'euc_dist_same', 'euc_dist_diff', 'euc_dist_same_std', 'euc_dist_diff_std'])

	for X_seed in X_seed_for_classes:
		#get seed class
		X_augment = pd.concat([X_seed]*num_queries)
		if model_type == 'dnn':
			_, seed_c = torch.max(model(torch.from_numpy(X_augment.values).float()).data, 1)
			seed_c = seed_c.numpy()
		else:
			seed_c = model.predict(X_augment)
		df_pred = pd.DataFrame.from_dict({'index': X_augment.index, 'seed_class': seed_c})

		#augment and add noise
		rand_noise = np.random.uniform(-1,1,size=X_augment.shape[0]*X_augment.shape[1]).reshape(X_augment.shape[0],-1)
		X_augment = X_augment + rand_noise

		#predict using model and assess class
		if model_type == 'dnn':
			_, c = torch.max(model(torch.from_numpy(X_augment.values).float()).data, 1)
			c = c.numpy()
		else:
			c = model.predict(X_augment)
		df_pred['noise_class'] = c

		#assess cosine similarity in queries
		adv_groups = df_pred.groupby(X_augment.index)
		for key,grp in adv_groups:
			same_class_indices = grp[grp['noise_class'] == grp['seed_class']].index.tolist()
			diff_class_indices = grp[grp['noise_class'] != grp['seed_class']].index.tolist()
			num_same_class = (grp['noise_class'] == grp['seed_class']).sum()

			#compute dist metrics
			df_cos_sim_same = np.array(cosine_similarity(X_augment.iloc[same_class_indices])) if len(same_class_indices) != 0 else np.zeros((len(adv_groups), len(adv_groups)))
			df_cos_sim_diff = np.array(cosine_similarity(X_augment.iloc[diff_class_indices])) if len(diff_class_indices) != 0 else np.zeros((len(adv_groups), len(adv_groups)))

			df_euc_dist_same = np.array(euclidean_distances(X_augment.iloc[same_class_indices])) if len(same_class_indices) != 0 else np.zeros((len(adv_groups), len(adv_groups)))
			df_euc_dist_diff = np.array(euclidean_distances(X_augment.iloc[diff_class_indices])) if len(diff_class_indices) != 0 else np.zeros((len(adv_groups), len(adv_groups)))

			#get all values above diagonal (since they repeat)
			np.fill_diagonal(df_cos_sim_same, 0)
			np.fill_diagonal(df_cos_sim_diff, 0)
			np.fill_diagonal(df_euc_dist_same, 0)
			np.fill_diagonal(df_euc_dist_diff, 0)

			df_cos_sim_same = np.triu(df_cos_sim_same, k=0)
			df_cos_sim_diff = np.triu(df_cos_sim_diff, k=0)
			df_euc_dist_same = np.triu(df_euc_dist_same, k=0)
			df_euc_dist_diff = np.triu(df_euc_dist_diff, k=0)

			df_out = df_out.append({'model_type': model_type, \
									'num_query': num_queries, \
									'seed_class': seed_c[0], \
									'num_same_class': num_same_class, \
									'cos_sim_same': np.mean(df_cos_sim_same[np.nonzero(df_cos_sim_same)]), \
									'cos_sim_diff': np.mean(df_cos_sim_diff[np.nonzero(df_cos_sim_diff)]), \
									'cos_sim_same_std': np.std(df_cos_sim_same[np.nonzero(df_cos_sim_same)]), \
									'cos_sim_diff_std': np.std(df_cos_sim_diff[np.nonzero(df_cos_sim_diff)]), \
									'euc_dist_same': np.mean(df_euc_dist_same[np.nonzero(df_euc_dist_same)]), \
									'euc_dist_diff': np.mean(df_euc_dist_diff[np.nonzero(df_euc_dist_diff)]), \
									'euc_dist_same_std': np.std(df_euc_dist_same[np.nonzero(df_euc_dist_same)]), \
									'euc_dist_diff_std': np.std(df_euc_dist_diff[np.nonzero(df_euc_dist_diff)])}, ignore_index=True)

	return df_out

def attack_db_query_distribution(model, X_seed_for_classes, pca, model_type='rf', num_queries=500):
	df_out = pd.DataFrame(columns=['model_type', 'num_query', 'seed_class', 'num_same_class', \
							'type', 'pca_c1', 'pca_c2'])

	for X_seed in X_seed_for_classes:
		#get seed class
		X_augment = pd.concat([X_seed]*num_queries)
		if model_type == 'dnn':
			_, seed_c = torch.max(model(torch.from_numpy(X_augment.values).float()).data, 1)
			seed_c = seed_c.numpy()
		else:
			seed_c = model.predict(X_augment)
		df_pred = pd.DataFrame.from_dict({'index': X_augment.index, 'seed_class': seed_c})

		#augment and add noise
		rand_noise = np.random.uniform(-1,1,size=X_augment.shape[0]*X_augment.shape[1]).reshape(X_augment.shape[0],-1)
		X_augment = X_augment + rand_noise

		#predict using model and assess class
		if model_type == 'dnn':
			_, c = torch.max(model(torch.from_numpy(X_augment.values).float()).data, 1)
			c = c.numpy()
		else:
			c = model.predict(X_augment)
		df_pred['noise_class'] = c

		#assess cosine similarity in queries
		adv_groups = df_pred.groupby(X_augment.index)
		for key,grp in adv_groups:
			same_class_indices = grp[grp['noise_class'] == grp['seed_class']].index.tolist()
			diff_class_indices = grp[grp['noise_class'] != grp['seed_class']].index.tolist()
			num_same_class = (grp['noise_class'] == grp['seed_class']).sum()

			#2d PCA
			X_same_decomposed = pca.transform(X_augment.iloc[same_class_indices])
			# X_diff_decomposed = pca.transform(X_augment.iloc[diff_class_indices])

			#merge into df
			df_same = pd.DataFrame(X_same_decomposed, columns=['pca_c1', 'pca_c2'])
			df_same['model_type'] = model_type
			df_same['seed_class'] = seed_c[0]
			df_same['num_query'] = num_queries
			df_same['type'] = 'same'
			df_same['num_same_class'] = num_same_class

			# df_diff = pd.DataFrame(X_diff_decomposed, columns=['pca_c1', 'pca_c2'])
			# df_diff['model_type'] = model_type
			# df_diff['seed_class'] = seed_c[0]
			# df_diff['num_query'] = num_queries
			# df_diff['type'] = 'diff'
			# df_diff['num_same_class'] = num_same_class

			df_out = pd.concat([df_out, df_same], ignore_index=True)

	return df_out

def attack_db_query(model, X_seed_for_classes, model_type='rf'):
	df_out = pd.DataFrame(columns=['model_type', 'num_query', 'seed_class', 'same_class_mean', 'same_class_std'])

	for query_size in [10,20,50,100,250,500,750,1000,2000,5000]:
		for X_seed in X_seed_for_classes:
			#get seed class
			X_augment = pd.concat([X_seed]*query_size)
			if model_type == 'dnn':
				_, seed_c = torch.max(model(torch.from_numpy(X_augment.values).float()).data, 1)
				seed_c = seed_c.numpy()
			else:
				seed_c = model.predict(X_augment)
			df_pred = pd.DataFrame.from_dict({'index': X_augment.index, 'seed_class': seed_c})

			#augment and add noise
			rand_noise = np.random.uniform(-1,1,size=X_augment.shape[0]*X_augment.shape[1]).reshape(X_augment.shape[0],-1)
			X_augment = X_augment + rand_noise

			#predict using model and assess class
			if model_type == 'dnn':
				_, c = torch.max(model(torch.from_numpy(X_augment.values).float()).data, 1)
			else:
				c = model.predict(X_augment)
			df_pred['noise_class'] = c

			#assess cosine similarity in queries
			num_same_classes = []
			adv_groups = df_pred.groupby(X_augment.index)
			for key,grp in adv_groups:
				num_same_classes.append((grp['noise_class'] == grp['seed_class']).sum())

			df_out = df_out.append({'model_type': model_type, \
								'num_query': query_size, \
								'seed_class': seed_c[0], \
								'same_class_mean': np.mean(num_same_classes), \
								'same_class_std': np.std(num_same_classes)}, ignore_index=True)
			
	return df_out

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-data_name', type=str, default='UCI_HAR')
	parser.add_argument('-num_users', type=int, default=100)
	parser.add_argument('-model_type', type=str, default='rf', help='rf, lr or dnn')
	parser.add_argument('-exp_num_query', type=bool)
	parser.add_argument('-exp_query_distance', type=bool)
	parser.add_argument('-exp_query_distribution', type=bool)
	args = parser.parse_args()

	X_train = pd.read_csv('./data/{}/train/X_train.txt'.format(args.data_name), delim_whitespace=True, header=None)
	X_test = pd.read_csv('./data/{}/test/X_test.txt'.format(args.data_name), delim_whitespace=True, header=None)
	y_test = pd.read_csv('./data/{}/test/y_test.txt'.format(args.data_name), delim_whitespace=True, header=None).squeeze()
	y_test = y_test-1

	#load model
	if args.model_type == 'rf':
		with open('./models/{}/rf.pkl'.format(args.data_name), 'rb') as f:
			model = pkl.load(f)
	elif args.model_type == 'lr':
		with open('./models/{}/lr.pkl'.format(args.data_name), 'rb') as f:
			model = pkl.load(f)
	elif args.model_type == 'dnn':
		model = torch.load('./models/{}/dnn.pt'.format(args.data_name))

	#pick random adversarial users
	random.seed(10)
	seed_for_classes = []
	if args.model_type == 'dnn':
		_, seed_c = torch.max(model(torch.from_numpy(X_test.values).float()).data, 1)
		seed_c = seed_c.numpy()
	else:
		seed_c = model.predict(X_test)
	
	for c in y_test.unique():
		c_indices = np.where(seed_c == c)[0].tolist()
		rand_nums = random.sample(c_indices, args.num_users)

		X_rand_users = X_test.iloc[rand_nums]
		seed_for_classes.append(X_rand_users)		

	## DB ATTACK
	if args.exp_num_query:
		attack_results = attack_db_query(model, seed_for_classes, args.model_type)
		if os.path.isfile('./results/{}/attack/attack_db_query.csv'.format(args.data_name)):
			df_results = pd.read_csv('./results/{}/attack/attack_db_query.csv'.format(args.data_name))
			df_results = df_results.append(attack_results, ignore_index=True)
			df_results.to_csv('./results/{}/attack/attack_db_query.csv'.format(args.data_name), index=False)
		else:
			attack_results.to_csv('./results/{}/attack/attack_db_query.csv'.format(args.data_name), index=False)

	if args.exp_query_distance:
		attack_results = attack_db_query_distance(model, seed_for_classes, args.model_type)
		if os.path.isfile('./results/{}/attack/attack_db_query_dist.csv'.format(args.data_name)):
			df_results = pd.read_csv('./results/{}/attack/attack_db_query_dist.csv'.format(args.data_name))
			df_results = df_results.append(attack_results, ignore_index=True)
			df_results.to_csv('./results/{}/attack/attack_db_query_dist.csv'.format(args.data_name), index=False)
		else:
			attack_results.to_csv('./results/{}/attack/attack_db_query_dist.csv'.format(args.data_name), index=False)

	if args.exp_query_distribution:
		pca = PCA(n_components=2)
		pca.fit(X=X_train)
		attack_results = attack_db_query_distribution(model, seed_for_classes, pca, args.model_type)
		if os.path.isfile('./results/{}/attack/attack_db_query_distribution.csv'.format(args.data_name)):
			df_results = pd.read_csv('./results/{}/attack/attack_db_query_distribution.csv'.format(args.data_name))
			df_results = df_results.append(attack_results, ignore_index=True)
			df_results.to_csv('./results/{}/attack/attack_db_query_distribution.csv'.format(args.data_name), index=False)
		else:
			attack_results.to_csv('./results/{}/attack/attack_db_query_distribution.csv'.format(args.data_name), index=False)

if __name__ == '__main__':
	main()