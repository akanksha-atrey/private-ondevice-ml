import os
import io
import sys
import time
from hwcounter import count, count_end

import onnxruntime as rt
import numpy as np
import pandas as pd

import nacl.secret

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Encryption key (generated via nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE))
key = b'$\xf4\x01A\x93\x8ce\xb0?\xba\xfa\x9e\x9925O\xdf\xd6\x1b~\xccc\xfb\x8cK=\xaf/\xcd\xdb\x07\xa5'

# Thresholds
alpha = 0.33
beta = 0.33
gamma = 0.33
detector_threshold = 0.2

# Path to the bundled ONNX model file
rf_model_path = './models/rf_encrypted.onnx'
lr_model_path = './models/lr_encrypted.onnx'
dnn_model_path = './models/dnn_quantized_encrypted.onnx'

def run_inference(X, model_rt_session, model_type):
    # Perform inference using the loaded ONNX model
    input_name = model_rt_session.get_inputs()[0].name
    output = model_rt_session.run(None, {input_name: X.astype(np.float32)})[0]

    if model_type == 'dnn':
        output = output.argmax(axis = 1)

    return output[0]

def encrypt_file(df, file_path):
    nonce = os.urandom(nacl.secret.SecretBox.NONCE_SIZE)
    box = nacl.secret.SecretBox(key)

    plaintext = df.to_csv(index=False).encode('utf-8')
    ciphertext = box.encrypt(plaintext, nonce)
    with open(file_path, 'wb') as file:
        file.write(ciphertext)

def decrypt_file(file_path):
    box = nacl.secret.SecretBox(key)

    with open(file_path, 'rb') as file:
        ciphertext = file.read()

    plaintext = box.decrypt(ciphertext)
    decrypted_df = pd.read_csv(io.StringIO(plaintext.decode('utf-8')))

    return decrypted_df

def decrypt_model(model_path):
    box = nacl.secret.SecretBox(key)
    
    with open(model_path, 'rb') as file:
        ciphertext = file.read()

    return box.decrypt(ciphertext)

def main(model_type):
    start_cycles = count()
    start_time = time.time()

    # Read input and other resource files
    X = np.genfromtxt('./data/input.txt').reshape(1, -1)

    # Load the ONNX model with ONNX Runtime
    if model_type == 'rf':
        model_ort_session = rt.InferenceSession(decrypt_model(rf_model_path))
    elif model_type == 'lr':
        model_ort_session = rt.InferenceSession(decrypt_model(lr_model_path))
    elif model_type == 'dnn':
        model_ort_session = rt.InferenceSession(decrypt_model(dnn_model_path))
    else:
        print("Error: Model type is not RF, LR, or DNN.")
        sys.exit(1)

    # Run inference
    output = run_inference(X, model_ort_session, model_type)
    print('Inference result: {}'.format(output))

    # Add runtime and cycles
    total_time = time.time() - start_time
    total_cycles = count_end() - start_cycles
    with open('./data/runtime.txt', 'a') as f:
        f.write('{}, {}\n'.format(total_time, total_cycles))

if __name__ == '__main__':
    # Get the input file from command-line arguments
    if len(sys.argv) != 2:
        print("Error: The <model_type> command-line arugment is required.")
        sys.exit(1)
    model_type = sys.argv[1]

    # Run the main function
    main(model_type)