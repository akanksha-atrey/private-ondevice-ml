'''
Author: Akanksha Atrey
Description: This file contains implementation to optimize pre-trained ML models for inference on 
			low-resource devices. The implementation support scikit-learn and PyTorch models.
'''

from src.defense.defense_training import Autoencoder, Net

import sys
import os
import pickle as pkl
import pandas as pd
import numpy as np
import time

import torch

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnx import numpy_helper, load

import nacl.secret

# Encryption key (generated via nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE))
key = b'$\xf4\x01A\x93\x8ce\xb0?\xba\xfa\x9e\x9925O\xdf\xd6\x1b~\xccc\xfb\x8cK=\xaf/\xcd\xdb\x07\xa5'

def sklearn_to_onnx(model, input_size, save_path):
	initial_type = [('float_input', FloatTensorType([None, input_size]))]
	onnx_model = convert_sklearn(model, initial_types=initial_type)
	with open(save_path, "wb") as f:
		f.write(onnx_model.SerializeToString())

def onnx_sklearn_inference(onnx_model_path, skmodel, X, y):
	sess = rt.InferenceSession(onnx_model_path)
	input_name = sess.get_inputs()[0].name

	onnx_pred = []
	onnx_time = []
	sk_pred = []
	sk_time = []
	
	for i in range(100):
		x = X[i].reshape(1,-1)

		start_time = time.time()
		pred_y = sess.run(None, {input_name: x.astype(np.float32)})[0]
		onnx_total_time = time.time() - start_time
		onnx_pred.append(pred_y[0])
		onnx_time.append(onnx_total_time)

		start_time = time.time()
		pred_y = skmodel.predict(x)
		sk_total_time = time.time() - start_time
		sk_pred.append(pred_y[0])
		sk_time.append(sk_total_time)

	
	onnx_perf = sum(onnx_pred == y[:100]) / 100
	sk_perf = sum(sk_pred == y[:100]) / 100

	print('ONNX Performance: ', onnx_perf)
	print('ONNX Inference time: ', sum(onnx_time)/len(onnx_time))

	print('Sklearn Performance: ', sk_perf)
	print('Sklearn Inference time: ', sum(sk_time)/len(sk_time))

def pytorch_to_onnx(model, input_size, onnx_save_path, quantized_save_path):
	dummy_input = torch.randn(1, input_size, requires_grad=True)

	torch.onnx.export(model,                       	  # model being run
		dummy_input, 		  						  # model input (or a tuple for multiple inputs)
		onnx_save_path,                               # where to save the model
		verbose=False,                
		opset_version=11,                             # the ONNX version to export the model to
	    do_constant_folding=True,                     # whether to execute constant folding for optimization
		input_names=['input'],                        # model's input names
		output_names=['output'],                      # model's output names
		dynamic_axes={'input' : {0: 'batch_size'},	  # variable length axes
						'output' : {0 : 'batch_size'}}
		)
	
	quantized_model = quantize_dynamic(onnx_save_path, quantized_save_path)

def onnx_pytorch_inference(onnx_model_path, pymodel, X, y, classification = True):
	pymodel.eval()
	sess = rt.InferenceSession(onnx_model_path)
	input_name = sess.get_inputs()[0].name

	onnx_pred = []
	onnx_time = []
	py_pred = []
	py_time = []
	
	for i in range(100):
		x = X[i].reshape(1,-1)

		start_time = time.time()
		pred_y = sess.run(None, {input_name: x.astype(np.float32)})[0]
		onnx_total_time = time.time() - start_time
		onnx_pred.append(pred_y)
		onnx_time.append(onnx_total_time)

		start_time = time.time()
		pred_y = pymodel(torch.tensor(x, dtype=torch.float32))[0]
		py_total_time = time.time() - start_time
		py_pred.append(pred_y.detach().numpy())
		py_time.append(py_total_time)

	if classification:
		onnx_perf = sum(np.array(onnx_pred).reshape(100, -1).argmax(axis = 1) == y[:100]) / 100
		py_perf = sum(np.array(py_pred).argmax(axis = 1) == y[:100]) / 100
	else:
		onnx_perf = mean_squared_error(X[:100], np.array(onnx_pred).reshape(100, -1))
		py_perf = mean_squared_error(X[:100], np.array(py_pred).reshape(100, -1))

	print('ONNX Performance: ', onnx_perf)
	print('ONNX Inference time: ', sum(onnx_time)/len(onnx_time))

	print('PyTorch Performance: ', py_perf)
	print('PyTorch Inference time: ', sum(py_time)/len(py_time))

def post_training_quantization(model, X, save_path):
	## note this function does only quantization. PTQ -> ONNX is not supported. Hence we go ONNX -> PTQ
	model.eval()

	# enable quantization for model
	quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
	
	# compare the quantized model with regular model
	start_time = time.time()
	X_decoded_vanilla, _ = model(torch.tensor(X.values, dtype=torch.float32))
	print(f'AE Vanilla MSE: {mean_squared_error(X, X_decoded_vanilla.detach().numpy())}')
	print(f'AE Vanilla Latency: {time.time() - start_time}')

	start_time = time.time()
	X_decoded_quantized, _ = quantized_model(torch.tensor(X.values, dtype=torch.float32))
	print(f'AE Quantized MSE: {mean_squared_error(X, X_decoded_quantized.detach().numpy())}')
	print(f'AE Quantized Latency: {time.time() - start_time}')

	print(model)
	print(quantized_model)

	# save model
	torch.jit.save(torch.jit.script(quantized_model), save_path)

	return quantized_model

def encrypt_model(model_path, encrypted_model_path):
    onnx_model = load(model_path)
    model_bytes = onnx_model.SerializeToString()
    
    nonce = os.urandom(nacl.secret.SecretBox.NONCE_SIZE)
    box = nacl.secret.SecretBox(key)
    ciphertext = box.encrypt(model_bytes, nonce)

    with open(encrypted_model_path, 'wb') as file:
        file.write(ciphertext)

def main(system_type):

	if system_type == 'defense':
		model_folder = 'attack_defense'
	elif system_type == 'baseline':
		model_folder = 'attack'
	else:
		print('Error: System type needs to be provided (defense or baseline mode).')
		sys.exit(1)

	# Load dataset
	X_test = pd.read_csv('./data/UCI_HAR/test/X_test.txt', delim_whitespace=True, header=None)
	y_test = pd.read_csv('./data/UCI_HAR/test/y_test.txt', delim_whitespace=True, header=None).squeeze()
	y_test = y_test-1
	print("Test dataset shapes: {}, {}".format(X_test.shape, y_test.shape))
	
	if system_type == 'defense':
		### CONVERT AE
		print('-------------AE-------------')
		ae_model = torch.load('./models/UCI_HAR/{}/autoencoder.pt'.format(model_folder))
		# save_path = './models/UCI_HAR/prototype/ae_quantized.pt'
		# ae_quantized_model = post_training_quantization(ae_model, X_test, save_path)

		onnx_save_path = './models/UCI_HAR/{}/ae.onnx'.format(model_folder)
		quantized_save_path = './models/UCI_HAR/{}/ae_quantized.onnx'.format(model_folder)
		pytorch_to_onnx(ae_model, X_test.shape[1], onnx_save_path, quantized_save_path)

		# inference using onnx runtime
		onnx_pytorch_inference(quantized_save_path, ae_model, X_test.values, X_test.values, classification = False)

		sess = rt.InferenceSession(quantized_save_path)
		input_name = sess.get_inputs()[0].name
		X_encoded = sess.run(None, {input_name: X_test.values.astype(np.float32)})[1]

		# save encrypted file
		encrypt_model(quantized_save_path, './models/UCI_HAR/{}/ae_quantized_encrypted.onnx'.format(model_folder))
	else:
		X_encoded = X_test.to_numpy()


	### CONVERT RF
	print('-------------RF-------------')
	with open('./models/UCI_HAR/{}/rf.pkl'.format(model_folder), 'rb') as f:
		model = pkl.load(f)
	save_path = './models/UCI_HAR/{}/rf.onnx'.format(model_folder)
	sklearn_to_onnx(model, X_encoded.shape[1], save_path)

	# inference using onnx runtime
	onnx_sklearn_inference(save_path, model, X_encoded, y_test)

	# save encrypted file
	encrypt_model(save_path, './models/UCI_HAR/{}/rf_encrypted.onnx'.format(model_folder))
	
	sess = rt.InferenceSession(save_path)
	input_name = sess.get_inputs()[0].name
	pred = sess.run(None, {input_name: X_encoded.astype(np.float32)})[0]
	print('RF Accuracy Test All: ', sum(pred == y_test) / len(y_test))

	### CONVERT LR
	print('-------------LR-------------')
	with open('./models/UCI_HAR/{}/lr.pkl'.format(model_folder), 'rb') as f:
		model = pkl.load(f)
	save_path = './models/UCI_HAR/{}/lr.onnx'.format(model_folder)
	sklearn_to_onnx(model, X_encoded.shape[1], save_path)

	# inference using onnx runtime
	onnx_sklearn_inference(save_path, model, X_encoded, y_test)

	# save encrypted file
	encrypt_model(save_path, './models/UCI_HAR/{}/lr_encrypted.onnx'.format(model_folder))

	sess = rt.InferenceSession(save_path)
	input_name = sess.get_inputs()[0].name
	pred = sess.run(None, {input_name: X_encoded.astype(np.float32)})[0]
	print('LR Accuracy Test All: ', sum(pred == y_test) / len(y_test))


	### CONVERT DNN
	print('-------------DNN-------------')
	model = torch.load('./models/UCI_HAR/{}/dnn.pt'.format(model_folder))
	onnx_save_path = './models/UCI_HAR/{}/dnn.onnx'.format(model_folder)
	quantized_save_path = './models/UCI_HAR/{}/dnn_quantized.onnx'.format(model_folder)
	pytorch_to_onnx(model, X_encoded.shape[1], onnx_save_path, quantized_save_path)

	# inference using onnx runtime
	onnx_pytorch_inference(quantized_save_path, model, X_encoded, y_test)

	# save encrypted file
	encrypt_model(quantized_save_path, './models/UCI_HAR/{}/dnn_quantized_encrypted.onnx'.format(model_folder))

	sess = rt.InferenceSession(quantized_save_path)
	input_name = sess.get_inputs()[0].name
	pred = sess.run(None, {input_name: X_encoded.astype(np.float32)})[0]
	print('DNN Accuracy Test All: ', sum(np.array(pred).reshape(-1, 6).argmax(axis = 1) == y_test) / len(y_test))

if __name__ == '__main__':
	main(sys.argv[1])