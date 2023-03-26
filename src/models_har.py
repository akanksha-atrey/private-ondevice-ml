import os
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

def decision_tree(X_train, y_train, X_test, y_test):
	clf = DecisionTreeClassifier(random_state=7)
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	print(f'RF test accuracy: {accuracy:.2f}')

	return clf

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

def dnn(train_loader, input_size, num_classes):
	model = Net(input_size, num_classes)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	model.train()

	# Train the model for 10 epochs
	for epoch in range(10):
		for x,y in train_loader:
			outputs = model(x)
			loss = criterion(outputs, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	return model

def dnn_predict(model, X_test, y_test):
	model.eval()
	outputs = model(X_test)
	_, predicted = torch.max(outputs.data, 1)
	accuracy = (predicted == y_test).float().mean()

	print("DNN test set accuracy: {:.2f}".format(accuracy))

def main():
	X_train = pd.read_csv('./data/UCI_HAR/train/X_train.txt', delim_whitespace=True, header=None)
	y_train = pd.read_csv('./data/UCI_HAR/train/y_train.txt', delim_whitespace=True, header=None).squeeze()
	X_test = pd.read_csv('./data/UCI_HAR/test/X_test.txt', delim_whitespace=True, header=None)
	y_test = pd.read_csv('./data/UCI_HAR/test/y_test.txt', delim_whitespace=True, header=None).squeeze()

	y_train = y_train-1
	y_test = y_test-1

	print("Train dataset shapes: {}, {}".format(X_train.shape, y_train.shape))
	print("Test dataset shapes: {}, {}".format(X_test.shape, y_test.shape))

	if not os.path.exists('./models/UCI_HAR'):
		os.makedirs('./models/UCI_HAR')

	#train lr
	lr_model = logistic_regression(X_train, y_train, X_test, y_test)
	with open('./models/UCI_HAR/activity_recognition/lr.pkl', 'wb') as f:
		pkl.dump(lr_model, f)

	#train rf
	rf_model = random_forest(X_train, y_train, X_test, y_test)
	with open('./models/UCI_HAR/activity_recognition/rf.pkl', 'wb') as f:
		pkl.dump(rf_model, f)

	#train dt
	dt_model = decision_tree(X_train, y_train, X_test, y_test)
	with open('./models/UCI_HAR/activity_recognition/dt.pkl', 'wb') as f:
		pkl.dump(dt_model, f)

	#train dnn
	X_train = torch.from_numpy(X_train.values).float()
	y_train = torch.from_numpy(y_train.values).long()
	X_test = torch.from_numpy(X_test.values).float()
	y_test = torch.from_numpy(y_test.values).long()

	train_dataset = TensorDataset(X_train, y_train)
	train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
	test_dataset = TensorDataset(X_test, y_test)
	test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

	model = dnn(train_loader, X_train.shape[1], 6)
	dnn_predict(model, X_test, y_test)
	torch.save(model, './models/UCI_HAR/activity_recognition/dnn.pt')

if __name__ == '__main__':
	main()