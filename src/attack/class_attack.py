'''
Author: Akanksha Atrey
Description: This file contains implementation of attacks on trained ML models to leak class information.
'''

from src.models_har import Net
from src.ensemble_models_har import predict_weighted

import os
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

'''
Randomly query all model input features for differing query sample sizes.
'''
def attack_wb_rand_query_size(model, X, y, model_type='rf', ensemble=1):
	attack_results_random = pd.DataFrame(columns=['model_type', 'query_size', 'accuracy', 'runtime'])

	#run baseline attack
	start_time = time.time()
	if ensemble > 1:
		results_baseline, _, _ = predict_weighted(model, X, y, num_models=ensemble, model_type=model_type)
	else:
		if model_type == 'dnn':
			_, results_baseline = torch.max(model(torch.from_numpy(X.values).float()).data, 1)
		else:
			results_baseline = model.predict(X)
	df_pred = pd.DataFrame.from_dict({'index': X.index, 'pred': results_baseline})
	attack_acc = df_pred['pred'].groupby(X.index).nunique().mean()/6*100
	end_time = time.time() - start_time
	row = {'model_type': model_type, 'query_size': 1, 'accuracy': attack_acc, 'runtime': end_time}
	attack_results_random = attack_results_random.append(row, ignore_index=True)

	#run attack with differing query sizes
	for query_size in [10,20,50,100,250,500,750,1000,2000,5000]:
		print("Sample size: ", query_size)
		start_time = time.time()

		X_augment = pd.concat([X]*query_size)
		X_augment[X_augment.columns] = np.random.uniform(-1,1,size=len(X_augment)*X_augment.shape[1]).reshape(len(X_augment),-1)
		
		if ensemble > 1:
			result_attack, _, _ = predict_weighted(model, X_augment, pd.concat([y]*query_size), num_models=ensemble, model_type=model_type)
		else:
			if model_type == 'dnn':
				_, result_attack = torch.max(model(torch.from_numpy(X_augment.values).float()).data, 1)
			else:
				result_attack = model.predict(X_augment)
		df_pred = pd.DataFrame.from_dict({'index': X_augment.index, 'pred': result_attack})
		attack_acc = df_pred['pred'].groupby(X_augment.index).nunique().mean()/6*100
		attack_results_random = attack_results_random.append({'model_type': model_type, \
															'query_size': query_size, \
															'accuracy': attack_acc, \
															'runtime': end_time}, \
															ignore_index=True)
		
		end_time = time.time() - start_time
		print("Total time to run random data attack for {} samples: {}".format(query_size, end_time))
	
	return attack_results_random

'''
Attack to assess how non model input features affect attack performance (run time).
'''
def attack_bb_rand_query_size(model, X, y, model_type='rf', ensemble=1):
	df_out = pd.DataFrame(columns=['model_type', 'num_extra_features', 'query_size', 'accuracy', 'runtime'])
	X_cols = list(X.columns)

	for num_extra_feat in [10,1000,5000]:
		for query_size in [10,20,50,100,250,500,750,1000,2000,5000]:
			start_time = time.time()

			#create random feature subset from extra features
			X_cols_augment = X_cols + list(range(X_cols[-1]+1, X_cols[-1]+num_extra_feat+1))
		
			#perturb subset
			X_augment = X.reindex(columns=X_cols_augment, fill_value=0)
			X_augment = pd.concat([X_augment]*query_size)
			X_augment[X_augment.columns] = np.random.default_rng().uniform(-1,1,size=(len(X_augment), X_augment.shape[1]))
			
			#get results
			if ensemble > 1:
				result_attack, _, _ = predict_weighted(model, X_augment[X_cols], pd.concat([y]*query_size), num_models=ensemble, model_type=model_type)
			else:
				if model_type == 'dnn':
					_, result_attack = torch.max(model(torch.from_numpy(X_augment[X_cols].values).float()).data, 1)
				else:
					result_attack = model.predict(X_augment[X_cols])
			df_pred = pd.DataFrame.from_dict({'index': X_augment.index, 'pred': result_attack})
			attack_acc = df_pred['pred'].groupby(X_augment.index).nunique().mean()/6*100

			end_time = time.time() - start_time
			print("Total time to run random data attack for {} samples: {}".format(query_size, end_time))

			df_out = df_out.append({'model_type': model_type, \
									'num_extra_features': num_extra_feat, \
									'query_size': query_size, \
									'accuracy': attack_acc, \
									'runtime': end_time}, ignore_index=True)

		estr = '_ensemble'+str(ensemble) if ensemble > 1 else ''
		df_out.to_csv('./results/{}/attack/attack{}_bb_rand_query_{}.csv'.format('UCI_HAR', estr, model_type), index=False)
	
	return df_out

'''
Randomly query different number of features for a given data set size.
'''
def attack_wb_rand_feature_size(model, X, y, model_type='rf', query_size=1000, num_samples=10, ensemble=1):
	attack_results_random = pd.DataFrame(columns=['model_type', 'num_features', 'accuracy_mean', 'accuracy_std', 'runtime'])

	#run baseline attack
	start_time = time.time()
	if ensemble > 1:
		results_baseline, _, _ = predict_weighted(model, X, y, num_models=ensemble, model_type=model_type)
	else:
		if model_type == 'dnn':
			_, results_baseline = torch.max(model(torch.from_numpy(X.values).float()).data, 1)
		else:
			results_baseline = model.predict(X)
	df_pred = pd.DataFrame.from_dict({'index': X.index, 'pred': results_baseline})
	attack_acc = df_pred['pred'].groupby(X.index).nunique().mean()/6*100
	end_time = time.time() - start_time
	row = {'model_type': model_type, 'num_features': 0, 'accuracy_mean': attack_acc, 'accuracy_std': 0, 'runtime': end_time}
	attack_results_random = attack_results_random.append(row, ignore_index=True)
		
	#run attack with different features queried for
	for num_features in [5,10,50,100,200,300,400,500]:#[5,10,20,30,40,50,60,70,80,90,100]:
		print("Number of features queried for: ", num_features)
		start_time = time.time()
		results_across_samples = []
		for i in range(num_samples):
			#randomly select columns
			random_cols = random.sample(list(X.columns), num_features)

			X_augment = pd.concat([X]*query_size)
			X_augment[random_cols] = np.random.uniform(-1,1,size=len(X_augment)*num_features).reshape(len(X_augment),-1)
			
			if ensemble > 1:
				result_attack, _, _ = predict_weighted(model, X_augment, pd.concat([y]*query_size), num_models=ensemble, model_type=model_type)
			else:
				if model_type == 'dnn':
					_, result_attack = torch.max(model(torch.from_numpy(X_augment.values).float()).data, 1)
				else:
					result_attack = model.predict(X_augment)

			df_pred = pd.DataFrame.from_dict({'index': X_augment.index, 'pred': result_attack})
			attack_acc = df_pred['pred'].groupby(X_augment.index).nunique().mean()/6*100
			results_across_samples.append(attack_acc)
		
		end_time = time.time() - start_time
		attack_results_random = attack_results_random.append({'model_type': model_type, \
															'num_features': num_features, \
															'accuracy_mean': np.mean(results_across_samples), \
															'accuracy_std': np.std(results_across_samples), \
															'runtime': end_time}, \
															ignore_index=True)

	return attack_results_random

'''
Attack to assess how non model input features affect attack performance (run time). TODO
'''
def attack_bb_query_extra(model, X, y, model_type='rf', ensemble=1):
	df_out = pd.DataFrame(columns=['model_type', 'num_extra_features', 'query_size', 'accuracy', 'runtime'])
	X_cols = list(X.columns)

	for num_extra_feat in [1,5,10,50,100,200,500,1000,2000,5000,10000]:
		for query_size in [10,1000,5000]:#[10,20,50,100,250,500,750,1000,2000,5000]:
			start_time = time.time()

			#create random feature subset from extra features
			X_cols_augment = X_cols + list(range(X_cols[-1]+1, X_cols[-1]+num_extra_feat+1))
		
			#perturb subset
			X_augment = X.reindex(columns=X_cols_augment, fill_value=0)
			X_augment = pd.concat([X_augment]*query_size)
			X_augment[X_augment.columns] = np.random.default_rng().uniform(-1,1,size=(len(X_augment), X_augment.shape[1]))
			
			#get output using black box inference
			if ensemble > 1:
				result_attack, _, _ = predict_weighted(model, X_augment[X_cols], pd.concat([y]*query_size), num_models=ensemble, model_type=model_type)
			else:
				if model_type == 'dnn':
					_, result_attack = torch.max(model(torch.from_numpy(X_augment[X_cols].values).float()).data, 1)
				else:
					result_attack = model.predict(X_augment[X_cols])
			df_pred = pd.DataFrame.from_dict({'index': X_augment.index, 'pred': result_attack})
			attack_acc = df_pred['pred'].groupby(X_augment.index).nunique().mean()/6*100

			runtime = time.time() - start_time
			df_out = df_out.append({'model_type': model_type, \
									'num_extra_features': num_extra_feat, \
									'query_size': query_size, \
									'accuracy': attack_acc, \
									'runtime': runtime}, ignore_index=True)

			estr = '_ensemble'+str(ensemble) if ensemble > 1 else ''
			df_out.to_csv('./results/{}/attack/attack{}_bb_unused_feat_{}.csv'.format('UCI_HAR', estr, model_type), index=False)

	return df_out

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-data_name', type=str, default='UCI_HAR')
	parser.add_argument('-num_users', type=int, default=100)
	parser.add_argument('-model_type', type=str, default='dt', help='rf, dt, lr, or dnn')
	parser.add_argument('-ensemble', default=1, type=int, help='1 if using single model, else number of models in ensemble')
	parser.add_argument('-wb_query_attack', type=bool)
	parser.add_argument('-wb_num_feat_attack', type=bool)
	parser.add_argument('-bb_query_attack', type=bool)
	parser.add_argument('-bb_unused_feat_attack', type=bool)
	args = parser.parse_args()

	X_test = pd.read_csv('./data/{}/test/X_test.txt'.format(args.data_name), delim_whitespace=True, header=None)
	y_test = pd.read_csv('./data/{}/test/y_test.txt'.format(args.data_name), delim_whitespace=True, header=None).squeeze()
	y_test = y_test-1

	#load model
	ensemble = '_ensemble'+str(args.ensemble) if args.ensemble > 1 else ''
	if args.model_type == 'rf':
		with open('./models/{}/activity_recognition/rf{}.pkl'.format(args.data_name, ensemble), 'rb') as f:
			model = pkl.load(f)
	elif args.model_type == 'lr':
		with open('./models/{}/activity_recognition/lr{}.pkl'.format(args.data_name, ensemble), 'rb') as f:
			model = pkl.load(f)
	elif args.model_type == 'dt':
		with open('./models/{}/activity_recognition/dt{}.pkl'.format(args.data_name, ensemble), 'rb') as f:
			model = pkl.load(f)
	elif args.model_type == 'dnn':
		model = torch.load('./models/{}/activity_recognition/dnn{}.pt'.format(args.data_name, ensemble))

	#pick 100 random adversarial users
	random.seed(10)
	rand_nums = random.sample(range(len(X_test)), args.num_users)
	X_rand_users = X_test.iloc[rand_nums]
	y_rand_users = y_test.iloc[rand_nums]

	## QUERY ATTACK
	#white-box
	if args.wb_query_attack:
		attack_results = attack_wb_rand_query_size(model, X_rand_users, y_rand_users, model_type=args.model_type, ensemble=args.ensemble)
		if os.path.isfile('./results/{}/attack/attack{}_rand_query.csv'.format(args.data_name, ensemble)):
			df_results = pd.read_csv('./results/{}/attack/attack{}_rand_query.csv'.format(args.data_name, ensemble))
			df_results = df_results.append(attack_results, ignore_index=True)
			df_results.to_csv('./results/{}/attack/attack{}_rand_query.csv'.format(args.data_name, ensemble), index=False)
		else:
			attack_results.to_csv('./results/{}/attack/attack{}_rand_query.csv'.format(args.data_name, ensemble), index=False)

	#black-box
	if args.bb_query_attack:
		attack_results = attack_bb_rand_query_size(model, X_rand_users, y_rand_users, model_type=args.model_type, ensemble=args.ensemble)

	## NUMBER OF FEATURES ATTACK (wb)
	if args.wb_num_feat_attack:
		attack_results = attack_wb_rand_feature_size(model, X_rand_users, y_rand_users, model_type=args.model_type, ensemble=args.ensemble)
		if os.path.isfile('./results/{}/attack/attack{}_rand_feat.csv'.format(args.data_name, ensemble)):
			df_results = pd.read_csv('./results/{}/attack/attack{}_rand_feat.csv'.format(args.data_name, ensemble))
			df_results = df_results.append(attack_results, ignore_index=True)
			df_results.to_csv('./results/{}/attack/attack{}_rand_feat.csv'.format(args.data_name, ensemble), index=False)
		else:
			attack_results.to_csv('./results/{}/attack/attack{}_rand_feat.csv'.format(args.data_name, ensemble), index=False)

	## NUMBER OF UNUSED FEATURES (bb)
	if args.bb_unused_feat_attack:
		attack_results = attack_bb_query_extra(model, X_rand_users, y_rand_users, model_type=args.model_type, ensemble=args.ensemble)

if __name__ == '__main__':
	main()