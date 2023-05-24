'''
Author: Akanksha Atrey
Description: This file contains implementation of an autoencoder based defense mechanism.
'''

import os
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import time
from hwcounter import count, count_end

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from skorch import NeuralNetClassifier, NeuralNetRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('text', usetex=False)
plt.rcParams['axes.labelsize']=16

DEVICE = None

class Autoencoder(nn.Module):
	def __init__(self, input_size, hidden_size=256, encoding_size=128):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, encoding_size),
			nn.ReLU()
		)
		self.decoder = nn.Sequential(
			nn.Linear(encoding_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, input_size)
		)
	
	def forward(self, x):
		out_encoder = self.encoder(x)
		out_decoder = self.decoder(out_encoder)
		return out_decoder, out_encoder
	
def cv_ae(X_train):
	class AutoEncoderNet(NeuralNetRegressor):
		def get_loss(self, y_pred, y_true, *args, **kwargs):
			decoded, _ = y_pred
			loss = super().get_loss(decoded, y_true, *args, **kwargs)
			return loss

	net = AutoEncoderNet(
		Autoencoder,
		max_epochs=10,
		lr=1e-3,
		iterator_train__shuffle=True,
		optimizer=torch.optim.AdamW,
		criterion=torch.nn.MSELoss,
		module__input_size=X_train.shape[1]
	)

	# deactivate skorch-internal train-valid split and verbose logging
	net.set_params(train_split=False, verbose=0)

	# set HPs to search over
	params = {
		'lr': [1e-3, 1e-5, 1e-7],
		'max_epochs': [10, 20],
		'batch_size': [128],
		'module__hidden_size': [128, 256, 512],
		'module__encoding_size': [64, 128]
	}
	rs = RandomizedSearchCV(net, params, n_iter=10, cv=3, refit=False, scoring='neg_mean_squared_error', verbose=2)
	rs.fit(X_train.to_numpy(dtype='float32'), X_train.to_numpy(dtype='float32'))
	print("best score: {:.3f}, best params: {}".format(rs.best_score_, rs.best_params_))

	return rs.best_params_
	
def train_ae(X_train, X_test, train_loader):
	# run CV
	cv_params = cv_ae(X_train)

	# Define the model, loss function and optimizer
	start_cycles = count()
	stime = time.time()

	model = Autoencoder(X_train.shape[1], cv_params['module__hidden_size'], cv_params['module__encoding_size']).to(device=DEVICE)
	criterion = nn.MSELoss()
	optimizer = optim.AdamW(model.parameters(), lr=cv_params['lr'])

	# Train the model
	model.train()
	for epoch in range(cv_params['max_epochs']):
		running_loss = 0.0
		for x,_ in train_loader:
			x = x.to(device=DEVICE)
			optimizer.zero_grad()
			output_decoder, _ = model(x)
			loss = criterion(output_decoder, x)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		print("Epoch {} Loss: {}".format(epoch+1, running_loss/X_train.shape[0]))

	elapsed = count_end() - start_cycles
	print(f'AE training time: {time.time()-stime}')
	print(f'AE training cycles: {elapsed}')

	# prediction statistics
	model.eval()
	X_decoded_train, _ = model(torch.tensor(X_train.values, dtype=torch.float32))
	print(f'AE MSE: {mean_squared_error(X_train, X_decoded_train.detach().numpy()):.3f}')
	X_decoded_test, _ = model(torch.tensor(X_test.values, dtype=torch.float32))
	print(f'AE MSE: {mean_squared_error(X_test, X_decoded_test.detach().numpy()):.3f}')

	return model

def train_rf(X_train, y_train, X_test, y_test):
	# run CV
	rf_grid = {"n_estimators": np.arange(10, 100, 10),
		   "min_samples_leaf": np.arange(1, 20, 2),
		   "max_features": [0.5, 1, "sqrt", "auto"]}

	rs = RandomizedSearchCV(RandomForestClassifier(random_state=7),
								param_distributions=rf_grid,
								n_iter=10,
								cv=3,
								verbose=2)
	rs.fit(X_train, y_train)
	print("best score: {:.3f}, best params: {}".format(rs.best_score_, rs.best_params_))

	# train model
	start_cycles = count()
	stime = time.time()

	clf = RandomForestClassifier(random_state=7, **rs.best_params_)
	clf.fit(X_train, y_train)

	elapsed = count_end() - start_cycles
	print(f'RF training time: {time.time()-stime}')
	print(f'RF training cycles: {elapsed}')

	# prediction statistics
	y_train_pred = clf.predict(X_train)
	print(f'RF train accuracy: {accuracy_score(y_train, y_train_pred):.2f}')
	print(f'RF train f1: {f1_score(y_train, y_train_pred, average="macro"):.2f}')
	y_test_pred = clf.predict(X_test)
	print(f'RF test accuracy: {accuracy_score(y_test, y_test_pred):.2f}')
	print(f'RF test f1: {f1_score(y_test, y_test_pred, average="macro"):.2f}')

	return clf

def train_lr(X_train, y_train, X_test, y_test):
	# run CV
	lr_grid = {"max_iter": np.arange(100, 500, 100),
		   "multi_class": ["multinomial"]}

	rs = RandomizedSearchCV(LogisticRegression(random_state=7),
								param_distributions=lr_grid,
								n_iter=10,
								cv=3,
								verbose=2)
	rs.fit(X_train, y_train)
	print("best score: {:.3f}, best params: {}".format(rs.best_score_, rs.best_params_))

	# train model
	start_cycles = count()
	stime = time.time()

	clf = LogisticRegression(random_state=7, **rs.best_params_)
	clf.fit(X_train, y_train)

	elapsed = count_end() - start_cycles
	print(f'LR training time: {time.time()-stime}')
	print(f'LR training cycles: {elapsed}')

	# prediction statistics
	y_train_pred = clf.predict(X_train)
	print(f'LR train accuracy: {accuracy_score(y_train, y_train_pred):.2f}')
	print(f'LR train f1: {f1_score(y_train, y_train_pred, average="macro"):.2f}')
	y_test_pred = clf.predict(X_test)
	print(f'LR test accuracy: {accuracy_score(y_test, y_test_pred):.2f}')
	print(f'LR test f1: {f1_score(y_test, y_test_pred, average="macro"):.2f}')

	return clf

class Net(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		x = nn.functional.relu(self.fc1(x))
		x = nn.functional.relu(self.fc2(x))
		x = self.fc3(x)
		return x
	
def cv_dnn(X_train, y_train):
	net = NeuralNetClassifier(
		Net,
		max_epochs=10,
		lr=1e-3,
		iterator_train__shuffle=True,
		optimizer=torch.optim.AdamW,
		criterion=torch.nn.CrossEntropyLoss,
		module__num_classes=y_train.nunique(),
		module__input_size=X_train.shape[1],
	)

	# deactivate skorch-internal train-valid split and verbose logging
	net.set_params(train_split=False, verbose=0)

	# set HPs to search over
	params = {
		'lr': [1e-3, 1e-5, 1e-7],
		'max_epochs': [10, 20, 30],
		'batch_size': [128],
		'optimizer__weight_decay': [1e-5,1e-7,1e-9],
		'module__hidden_size': [64, 128, 256]
	}
	rs = RandomizedSearchCV(net, params, n_iter=10, cv=3, refit=False, scoring='accuracy', verbose=2)
	rs.fit(X_train, y_train)
	print("best score: {:.3f}, best params: {}".format(rs.best_score_, rs.best_params_))

	return rs.best_params_

def train_dnn(X_train, y_train, X_test, y_test, num_classes):
	# run CV
	cv_params = cv_dnn(X_train, y_train)

	# Define network, optimizer, and criterion
	start_cycles = count()
	stime = time.time()

	model = Net(X_train.shape[1], cv_params['module__hidden_size'], y_train.nunique()).to(device=DEVICE)

	train_dataset = TensorDataset(X_train, torch.from_numpy(y_train.values).long())
	train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=cv_params['lr'], weight_decay=cv_params['optimizer__weight_decay'])
	model.train()

	# Train the model for 10 epochs
	losses = []
	for epoch in range(cv_params['max_epochs']):
		for x,y in train_loader:
			x = x.to(device=DEVICE)
			y = y.to(device=DEVICE)

			outputs = model(x)
			loss = criterion(outputs, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		losses.append(loss)

	# plt.plot(losses)
	# plt.show()

	elapsed = count_end() - start_cycles
	print(f'DNN training time: {time.time()-stime}')
	print(f'DNN training cycles: {elapsed}')
	
	# prediction statistics
	y_train_pred = dnn_predict(model, X_train, torch.from_numpy(y_train.values).long())
	print(f'DNN train accuracy: {accuracy_score(y_train, y_train_pred):.2f}')
	print(f'DNN train f1: {f1_score(y_train, y_train_pred, average="macro"):.2f}')
	y_test_pred = dnn_predict(model, X_test, torch.from_numpy(y_test.values).long())
	print(f'DNN test accuracy: {accuracy_score(y_test, y_test_pred):.2f}')
	print(f'DNN test f1: {f1_score(y_test, y_test_pred, average="macro"):.2f}')

	return model

def dnn_predict(model, X, y, debug=False):
	model.eval()
	outputs = model(X)
	_, predicted = torch.max(outputs.data, 1)

	if debug:
		sns.heatmap(confusion_matrix(y, predicted))
		plt.show()

	return predicted.numpy()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-data_name', type=str, default='UCI_HAR')
	parser.add_argument('-model_type', type=str, default='rf', help='rf, lr or dnn')
	args = parser.parse_args()

	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Load dataset
	X_train = pd.read_csv('./data/{}/train/X_train.txt'.format(args.data_name), delim_whitespace=True, header=None)
	y_train = pd.read_csv('./data/{}/train/y_train.txt'.format(args.data_name), delim_whitespace=True, header=None).squeeze()
	X_test = pd.read_csv('./data/{}/test/X_test.txt'.format(args.data_name), delim_whitespace=True, header=None)
	y_test = pd.read_csv('./data/{}/test/y_test.txt'.format(args.data_name), delim_whitespace=True, header=None).squeeze()

	y_train = y_train-1
	y_test = y_test-1

	print("Train dataset shapes: {}, {}".format(X_train.shape, y_train.shape))
	print("Test dataset shapes: {}, {}".format(X_test.shape, y_test.shape))

	# Load subject information for train and test sets
	subject_train = pd.read_csv('./data/{}/train/subject_train.txt'.format(args.data_name), delim_whitespace=True, header=None)
	subject_test = pd.read_csv('./data/{}/test/subject_test.txt'.format(args.data_name), delim_whitespace=True, header=None)

	print("Number of users in train set: ", subject_train.nunique()[0])
	print("Number of users in test set: ", subject_test.nunique()[0])

	# Convert to Tensor
	train_dataset = TensorDataset(torch.from_numpy(X_train.values).float(), torch.from_numpy(y_train.values).long())
	train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
	test_dataset = TensorDataset(torch.from_numpy(X_test.values).float(), torch.from_numpy(y_test.values).long())
	test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

	# Train models
	if os.path.isfile('./models/{}/attack_defense/autoencoder.pt'.format(args.data_name)):
		ae_model = torch.load('./models/{}/attack_defense/autoencoder.pt'.format(args.data_name))
	else:
		ae_model = train_ae(X_train, X_test, train_loader)
		torch.save(ae_model, './models/{}/attack_defense/autoencoder.pt'.format(args.data_name))
	_, out_encoder_train = ae_model(torch.from_numpy(X_train.values).float())
	_, out_encoder_test = ae_model(torch.from_numpy(X_test.values).float())

	if args.model_type == 'rf':
		model = train_rf(out_encoder_train.detach().numpy(), y_train, out_encoder_test.detach().numpy(), y_test)
		with open('./models/{}/attack_defense/rf.pkl'.format(args.data_name), 'wb') as f:
			pkl.dump(model, f)
	elif args.model_type == 'lr':
		model = train_lr(out_encoder_train.detach().numpy(), y_train, out_encoder_test.detach().numpy(), y_test)
		with open('./models/{}/attack_defense/lr.pkl'.format(args.data_name), 'wb') as f:
			pkl.dump(model, f)
	elif args.model_type == 'dnn':
		model = train_dnn(out_encoder_train.detach(), y_train, out_encoder_test.detach(), y_test, num_classes=y_train.nunique())
		torch.save(model, './models/{}/attack_defense/dnn.pt'.format(args.data_name))

if __name__ == '__main__':
	main()