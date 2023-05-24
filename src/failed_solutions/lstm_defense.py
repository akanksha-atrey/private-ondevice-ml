'''
Author: Akanksha Atrey
Description: This file contains implementation of an LSTM based defense to identify adversarial usage.
'''

import os
import random
import argparse
import numpy as np
import pandas as pd
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import matplotlib.pyplot as plt

DEVICE = None

### Dataset class to produce overlapping time-series sequences
class DatasetLSTM(Dataset):
	"""
		Support class for the loading and batching of sequences of samples

		Args:
			dataset (Tensor): Tensor containing all the samples
			sequence_length (int): length of the analyzed sequence by the LSTM
		Return:
			X: contains two time sequences
			y: contains location of following timestep
	"""

	##  Constructor
	def __init__(self, dataset, label, sequence_length=1):
		self.dataset = dataset
		self.label = label
		self.seq_length = sequence_length

	##  Override total dataset's length getter
	def __len__(self):
		return self.dataset.__len__()

	##  Override single items' getter to return overlapping sequences of size self.seq_length
	def __getitem__(self, idx):
		if idx + self.seq_length >= self.__len__():
			idx = self.__len__()-self.seq_length-1

		return torch.FloatTensor(self.dataset[idx:idx+self.seq_length].values), \
				torch.LongTensor([self.label])

class LSTM(nn.Module):
	def __init__(self, input_dim, hidden_dim, batch_size, dropout_p):
		super(LSTM, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size

		#layers
		self.lstm1 = nn.LSTM(self.input_dim, self.hidden_dim, 1, batch_first=True)
		self.dropout1 = nn.Dropout(p=dropout_p)
		# self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim, 1, batch_first=True)
		# self.dropout2 = nn.Dropout(p=dropout_p)
		self.linear_loc = nn.Linear(self.hidden_dim, 1)

	def forward(self, input_seq):
		# Forward pass through LSTM layer input_seq : (batch_size, seq_len, features)
		# shape of lstm_out: (batch_size, seq_len, features)
		# shape of self.hidden: (a, b), where a and b both have shape (num_layers, batch_size, hidden_dim).
	
		lstm1_out, _ = self.lstm1(input_seq.view(len(input_seq), -1, self.input_dim)) #size [batch_size, seq_length, hidden_dim]
		dropout1 = self.dropout1(lstm1_out)
		# lstm2_out, _ = self.lstm2(dropout1)
		# dropout2 = self.dropout2(lstm2_out)
		y_loc = self.linear_loc(dropout1[:,-1])

		return y_loc

def train_lstm(data_loader, input_size=4, hidden_dim=1, \
			epochs=10, lr=1e-2, batch_size=64, weight_decay=1e-6, dropout_p=0.1):

	model = LSTM(input_dim=input_size, hidden_dim=hidden_dim,  \
				batch_size=batch_size, dropout_p=dropout_p)
	optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	criterion = nn.BCEWithLogitsLoss()

	model = model.to(device=DEVICE)  # move the model parameters to CPU/GPU

	train_loss = []
	for e in range(epochs):
		for x, y in data_loader:
			model.train()

			x = x.to(device=DEVICE, dtype=torch.float32)
			y = y.to(device=DEVICE, dtype=torch.float32)

			scores = model(x)
			loss = criterion(scores, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		train_loss.append(loss.item())
		print('epoch: {}, loss: {}'.format(e, loss.item()))
	
	plt.plot(train_loss)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.show()

	return model

def predict(model, data_loader):
	y_test = []
	y_pred = []
	conf_max = []

	model.eval()
	with torch.no_grad():
		for i,(x,y) in enumerate(data_loader):
			scores = model(x)
			conf_max.extend(torch.sigmoid(scores).squeeze().tolist())
			y_out = torch.round(torch.sigmoid(scores))

			y_pred.extend(y_out.squeeze().tolist())
			y_test.extend(y.squeeze().tolist())
	print(len(y_pred), len(y_test))
	# print(y_pred)
	print(np.array(y_test).mean())
	i_benign = np.where(y_test == 0)[0]
	i_adv = np.where(y_test==1)[0]

	print('benign conf: ', np.array(conf_max)[i_benign].mean())
	print('adv conf: ', np.array(conf_max)[i_adv].mean())

	print((np.array(y_pred) == np.array(y_test)).sum())
	acc = (np.array(y_pred) == np.array(y_test)).sum()/float(len(y_test))

	return acc

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-filename', type=str)
	parser.add_argument('-seq_length', default=10, type=int, help="Lookback time period for each timestep.")
	parser.add_argument('-use_gpu', default=False, type=bool)
	parser.add_argument('-model_path', default=None, type=str)
	parser.add_argument('-debug', default=False, type=bool)
	parser.add_argument('-batch_size', default=64, type=int, help="LSTM batch size.")
	parser.add_argument('-epochs', default=10, type=int)
	parser.add_argument('-lr', default=1e-3, type=float)
	parser.add_argument('-weight_decay', default=1e-8, type=float)
	parser.add_argument('-dropout_p', default=0.1, type=float)
	parser.add_argument('-hidden_dim', default=128, type=int)
	args = parser.parse_args()

	if args.use_gpu and torch.cuda.is_available():
		DEVICE = torch.device('cuda')
	else:
		DEVICE = torch.device('cpu')
	print('Device: ', DEVICE)

	# read benign data
	X_train_benign = pd.read_csv('./data/UCI_HAR/train/X_train.txt', delim_whitespace=True, header=None)
	X_train_benign['label'] = 0
	X_test_benign = pd.read_csv('./data/UCI_HAR/test/X_test.txt', delim_whitespace=True, header=None)
	X_test_benign['label'] = 0

	print("Benign input: ", X_train_benign.shape, X_test_benign.shape)

	# prep simulated data of adversarial usage
	random.seed(7)
	X_train_adv = X_train_benign.iloc[random.sample(range(len(X_train_benign)), 100)]
	X_train_adv = pd.concat([X_train_adv]*70)
	X_train_adv[X_train_adv.columns] = np.random.uniform(-1,1,size=len(X_train_adv)*X_train_adv.shape[1]).reshape(len(X_train_adv),-1)
	
	random.seed(2)
	X_test_adv = X_test_benign.iloc[random.sample(range(len(X_test_benign)), 100)]
	X_test_adv = pd.concat([X_test_adv]*30)
	X_test_adv[X_test_adv.columns] = np.random.uniform(-1,1,size=len(X_test_adv)*X_test_adv.shape[1]).reshape(len(X_test_adv),-1)

	print("Adversarial input: ", X_train_adv.shape, X_test_adv.shape)

	# reshape and prep data for lstm input (batch first) + combine benign and adv usage
	train_dataset_benign = DatasetLSTM(X_train_benign, 0, args.seq_length)
	train_dataset_adv = DatasetLSTM(X_train_adv, 1, args.seq_length)
	train_dataset = ConcatDataset([train_dataset_benign, train_dataset_adv])
	train_data_loader = DataLoader(train_dataset, args.batch_size, shuffle=False)

	test_dataset_benign = DatasetLSTM(X_test_benign, 0, args.seq_length)
	test_dataset_adv = DatasetLSTM(X_test_adv, 1, args.seq_length)
	test_dataset = ConcatDataset([test_dataset_benign, test_dataset_adv])
	test_data_loader = DataLoader(test_dataset, len(test_dataset), shuffle=False)

	# train lstm
	if not os.path.isfile('./models/UCI_HAR/attack_defense/lstm.pt'):
		model = train_lstm(train_data_loader, input_size=X_train_benign.shape[1], \
							hidden_dim=args.hidden_dim, epochs=args.epochs, lr=args.lr, \
							batch_size=args.batch_size, weight_decay=args.weight_decay, \
							dropout_p=args.dropout_p)
		torch.save(model, './models/UCI_HAR/attack_defense/lstm.pt')
	else:
		model = torch.load('./models/UCI_HAR/attack_defense/lstm.pt')

	# test lstm
	acc = predict(model, train_data_loader)
	print('Train Accuracy: ', acc)
	acc = predict(model, test_data_loader)
	print('Test Accuracy: ', acc)

if __name__ == '__main__':
	main()