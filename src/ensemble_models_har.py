'''
Author: Akanksha Atrey
Description: This file contains implementation of ensemble models (bootstrapped) aggregated via random weights.
'''

import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

def train_rf(X_train, y_train, num_trees=10):
	clf = RandomForestClassifier(n_estimators=num_trees, random_state=7)
	clf.fit(X_train, y_train)

	return clf

def train_lr(X_train_bootstrapped, y_train_bootstrapped, num_models=10):
	models = []

	for n in range(num_models):
		clf = LogisticRegression(random_state=0, max_iter=500, multi_class='multinomial')
		clf.fit(X_train_bootstrapped[n], y_train_bootstrapped[n])
		models.append(clf)

	return models

def train_dnn(X_train_bootstrapped, y_train_bootstrapped, input_size, num_classes, num_models=10):
	models = []

	for n in range(num_models):
		model = Net(input_size, num_classes)

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr=0.01)
		model.train()

		train_dataset = TensorDataset(torch.from_numpy(X_train_bootstrapped[n].values).float(), \
										torch.from_numpy(y_train_bootstrapped[n].values).long())
		train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

		for epoch in range(10):
			for x,y in train_loader:
				outputs = model(x)
				loss = criterion(outputs, y)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
		
		models.append(model)
		
	return models

class Net(nn.Module):
	def __init__(self, input_size, num_classes):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input_size, 32)
		self.fc2 = nn.Linear(32, 32)
		self.fc3 = nn.Linear(32, num_classes)

	def forward(self, x):
		x = nn.functional.relu(self.fc1(x))
		x = nn.functional.relu(self.fc2(x))
		x = self.fc3(x)
		return x

def predict_weighted(model, X, y, weights=None, num_models=10, model_type='rf'):
	if weights is None:
		weights = np.random.uniform(size=num_models)
		weights = weights / weights.sum()
	
	pred_proba = []
	m_accuracies = []
	models = model.estimators_ if model_type == 'rf' else model
	for i,m in enumerate(models):
		if model_type != 'dnn':
			pred_proba.append(m.predict_proba(X))
			m_accuracies.append(m.score(X,y))
		else:
			m.eval()
			X_tensor = torch.from_numpy(X.values).float()
			outputs = m(X_tensor).data

			pred_proba.append(outputs.numpy())
			_, predicted = torch.max(outputs.data, 1)
			m_accuracies.append((predicted.numpy() == y).mean())	
	
	pred_weighted = np.average(np.array(pred_proba), axis=0, weights=weights).argmax(axis=1)
	accuracy = (pred_weighted == y).sum()/len(X)

	return pred_weighted, accuracy, m_accuracies

def predict_weighted_diverse(models, X, y, weights=None, num_models=10):
	if weights is None:
		weights = np.random.uniform(size=num_models*3)
		weights = weights / weights.sum()
	
	pred_proba = []
	m_accuracies = []
	model_types = ['rf', 'lr', 'dnn']
	for m,model_type in enumerate(model_types):
		mods = models[m]
		mods = mods.estimators_ if model_type == 'rf' else mods

		for i,m in enumerate(mods):
			if model_type != 'dnn':
				pred_proba.append(m.predict_proba(X))
				m_accuracies.append(m.score(X,y))
			else:
				m.eval()
				X_tensor = torch.from_numpy(X.values).float()
				outputs = m(X_tensor).data

				pred_proba.append(outputs.numpy())
				_, predicted = torch.max(outputs.data, 1)
				m_accuracies.append((predicted.numpy() == y).mean())	
		
	pred_weighted = np.average(np.array(pred_proba), axis=0, weights=weights).argmax(axis=1)
	accuracy = (pred_weighted == y).sum()/len(X)

	return pred_weighted, accuracy, m_accuracies

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-data_name', type=str, default='UCI_HAR', help='name of the dataset')
	parser.add_argument('-model_type', type=str, default='rf', help='rf, lr or dnn')
	parser.add_argument('-num_models', type=int, default=10, help='number of models in ensemble')
	parser.add_argument('-with_replacement', type=str, default='true', help='replace while resampling data')
	args = parser.parse_args()

	replace = True if args.with_replacement.lower() == 'true' else False
	
	# load data
	X_train = pd.read_csv('./data/{}/train/X_train.txt'.format(args.data_name), delim_whitespace=True, header=None)
	y_train = pd.read_csv('./data/{}/train/y_train.txt'.format(args.data_name), delim_whitespace=True, header=None).squeeze()
	X_test = pd.read_csv('./data/{}/test/X_test.txt'.format(args.data_name), delim_whitespace=True, header=None)
	y_test = pd.read_csv('./data/{}/test/y_test.txt'.format(args.data_name), delim_whitespace=True, header=None).squeeze()

	y_train = y_train-1
	y_test = y_test-1

	print("Train dataset shapes: {}, {}".format(X_train.shape, y_train.shape))
	print("Test dataset shapes: {}, {}".format(X_test.shape, y_test.shape))

	if not os.path.exists('./models/UCI_HAR'):
		os.makedirs('./models/UCI_HAR')

	# bootstrap data
	X_train_bootstrapped = []
	y_train_bootstrapped = []
	for n in range(args.num_models):
		resampled_ix = resample(range(len(X_train)), replace=replace, n_samples=int(len(X_train)/args.num_models), random_state=7)
		X_train_bootstrapped.append(X_train.iloc[resampled_ix])
		y_train_bootstrapped.append(y_train[resampled_ix])

	# train ensemble
	if args.model_type == 'rf':
		rf_model = train_rf(X_train, y_train, args.num_models)
		pred_out, acc, ind_acc = predict_weighted(rf_model, X_test, y_test, num_models=args.num_models, model_type=args.model_type)

		with open('./models/UCI_HAR/rf_ensemble{}.pkl'.format(args.num_models, replace), 'wb') as f:
			pkl.dump(rf_model, f)
	elif args.model_type == 'lr':
		lr_model = train_lr(X_train_bootstrapped, y_train_bootstrapped, args.num_models)
		pred_out, acc, ind_acc  = predict_weighted(lr_model, X_test, y_test, num_models=args.num_models, model_type=args.model_type)

		with open('./models/UCI_HAR/lr_ensemble{}.pkl'.format(args.num_models, replace), 'wb') as f:
			pkl.dump(lr_model, f)
	elif args.model_type == 'dnn':
		dnn_model = train_dnn(X_train_bootstrapped, y_train_bootstrapped, X_train.shape[1], 6, num_models=args.num_models)
		pred_out, acc, ind_acc = predict_weighted(dnn_model, X_test, y_test, num_models=args.num_models, model_type=args.model_type)

		torch.save(dnn_model, './models/UCI_HAR/dnn_ensemble{}.pt'.format(args.num_models, replace))

	# assess ensemble with diverse models
	with open('./models/UCI_HAR/rf_ensemble{}.pkl'.format(args.num_models), 'rb') as f:
		rf_model = pkl.load(f)
	with open('./models/UCI_HAR/lr_ensemble{}.pkl'.format(args.num_models), 'rb') as f:
		lr_model = pkl.load(f)
	dnn_model = torch.load('./models/UCI_HAR/dnn_ensemble{}.pt'.format(args.num_models))
	models = [rf_model, lr_model, dnn_model]

	_, acc, _ = predict_weighted(rf_model, X_test, y_test, num_models=args.num_models, model_type='rf')
	print("RF ACC: ", acc)
	_, acc, _ = predict_weighted(lr_model, X_test, y_test, num_models=args.num_models, model_type='lr')
	print("LR ACC: ", acc)
	_, acc, _ = predict_weighted(dnn_model, X_test, y_test, num_models=args.num_models, model_type='dnn')
	print("DNN ACC: ", acc)
	pred_out, acc, ind_acc = predict_weighted_diverse(models, X_test, y_test, num_models=args.num_models)
	print("DIVERSE ACC: ", acc)

	# save output
	mlist = [f'ensemble_m{i+1}' for i in range(args.num_models)]
	df_out = pd.DataFrame(zip([args.model_type]*args.num_models, mlist, ind_acc), \
						columns=['model_type', 'model_info', 'accuracy'])
	df_out = df_out.append({'model_type': args.model_type, 'model_info': 'ensemble', 'accuracy': acc}, ignore_index=True)

	if os.path.isfile('./results/{}/ensemble/accuracies_ensemble{}.csv'.format(args.data_name, args.num_models)):
		df_results = pd.read_csv('./results/{}/ensemble/accuracies_ensemble{}.csv'.format(args.data_name, args.num_models))
		df_results = df_results.append(df_out, ignore_index=True)
		df_results.to_csv('./results/{}/ensemble/accuracies_ensemble{}.csv'.format(args.data_name, args.num_models), index=False)
	else:
		df_out.to_csv('./results/{}/ensemble/accuracies_ensemble{}.csv'.format(args.data_name, args.num_models), index=False)

if __name__ == '__main__':
	main()
