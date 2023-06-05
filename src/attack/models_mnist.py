import os
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

def logistic_regression(X_train, y_train, X_test, y_test):
	clf = LogisticRegression(random_state=0, max_iter=500, multi_class='multinomial')
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	print(f'LR test accuracy: {accuracy:.2f}')

	return clf

def random_forest(X_train, y_train, X_test, y_test):
	clf = RandomForestClassifier(n_estimators=100, random_state=7)
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	print(f'RF test accuracy: {accuracy:.2f}')

	return clf

class Net(nn.Module):
	def __init__(self, input_size, num_classes):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input_size, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, num_classes)

	def forward(self, x):
		x = nn.functional.relu(self.fc1(x))
		x = nn.functional.relu(self.fc2(x))
		x = self.fc3(x)
		return x

def dnn(train_loader, input_size, num_classes):
	model = Net(input_size, num_classes)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-4)
	model.train()

	# Train the model for 10 epochs
	for epoch in range(10):
		for x,y in train_loader:
			outputs = model(x)
			loss = criterion(outputs, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
		print(loss.item())

	return model

def dnn_predict(model, X_test, y_test):
	model.eval()
	outputs = model(X_test)
	_, predicted = torch.max(outputs.data, 1)
	accuracy = (predicted == y_test).float().mean()

	print("DNN test set accuracy: {:.2f}".format(accuracy))

def main():
	X_train = pd.read_csv('./data/MNIST/mnist_train.csv', header = None)
	X_train, y_train = X_train.iloc[:,1:], X_train.iloc[:,0]
	X_test = pd.read_csv('./data/MNIST/mnist_test.csv', header = None)
	X_test, y_test = X_test.iloc[:,1:], X_test.iloc[:,0]

	num_classes = y_train.nunique()

	print("Train dataset shapes: {}, {}".format(X_train.shape, y_train.shape))
	print("Test dataset shapes: {}, {}".format(X_test.shape, y_test.shape))

	if not os.path.exists('./models/MNIST'):
		os.makedirs('./models/MNIST')

	# standardize dataset into [-1, 1] scale
	scaler = MinMaxScaler((-1, 1))
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	
	pd.DataFrame(X_train).to_csv('./data/MNIST/X_train.csv', index=False, header=None)
	pd.DataFrame(y_train).to_csv('./data/MNIST/y_train.csv', index=False, header=None)
	pd.DataFrame(X_test).to_csv('./data/MNIST/X_test.csv', index=False, header=None)
	pd.DataFrame(y_test).to_csv('./data/MNIST/y_test.csv', index=False, header=None)

	# train lr
	lr_model = logistic_regression(X_train, y_train, X_test, y_test)
	with open('./models/MNIST/attack/lr.pkl', 'wb') as f:
		pkl.dump(lr_model, f)

	# train rf
	rf_model = random_forest(X_train, y_train, X_test, y_test)
	with open('./models/MNIST/attack/rf.pkl', 'wb') as f:
		pkl.dump(rf_model, f)

	# train dnn
	X_train = torch.from_numpy(X_train).float()
	y_train = torch.from_numpy(y_train).long()
	X_test = torch.from_numpy(X_test).float()
	y_test = torch.from_numpy(y_test).long()

	train_dataset = TensorDataset(X_train, y_train)
	train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
	test_dataset = TensorDataset(X_test, y_test)
	test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

	model = dnn(train_loader, X_train.shape[1], num_classes)
	dnn_predict(model, X_test, y_test)
	torch.save(model, './models/MNIST/attack/dnn.pt')

if __name__ == '__main__':
	main()