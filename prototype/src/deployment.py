import os
import io
import sys
import time
import threading
from hwcounter import count, count_end

import onnxruntime as rt
import numpy as np
import pandas as pd

import nacl.secret

from scipy.spatial.distance import cdist
from scipy.stats import entropy

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
ae_model_path = './models/ae_quantized_encrypted.onnx'
rf_model_path = './models/rf_encrypted.onnx'
lr_model_path = './models/lr_encrypted.onnx'
dnn_model_path = './models/dnn_quantized_encrypted.onnx'

def run_inference(X, ae_rt_session, model_rt_session, model_type):
    # Perform inference using the loaded ONNX model
    input_name = ae_rt_session.get_inputs()[0].name
    X_decoded, X_encoded = ae_rt_session.run(None, {input_name: X.astype(np.float32)})
    
    input_name = model_rt_session.get_inputs()[0].name
    output = model_rt_session.run(None, {input_name: X_encoded.astype(np.float32)})[0]

    if model_type == 'dnn':
        output = output.argmax(axis = 1)

    return X_encoded, X_decoded, output[0]

def run_detector_dist(X_encoded, output, prior_queries, detector_components):
    # Measure difference in input to closest element in class c
    ed_median = 0
    if len(prior_queries) != 0:
        prior_queries_same_class = prior_queries[prior_queries['pred_c'] == output]

        if len(prior_queries_same_class) != 0:
             distances = cdist(X_encoded.reshape(1,-1), prior_queries_same_class.drop(columns=['pred_c']), metric='euclidean')
             ed_median = np.median(distances, axis=1)[0]

    # Save output
    encoding_df = pd.DataFrame(X_encoded, columns = ['Column_' + str(i) for i in range(X_encoded.shape[1])])
    encoding_df['pred_c'] = output
    prior_queries = encoding_df if len(prior_queries) == 0 else prior_queries.append(encoding_df, ignore_index = True)
    detector_components[0] = prior_queries
    detector_components[3] = ed_median

def run_detector_reconstruction(X, X_decoded, detector_components):
    # Measure reconstruction error (x vs ae(x)) to separate out anomalies/OOD queries
    mse = (X - X_decoded)**2
    detector_components[4] = np.mean(mse)

def run_detector_entropy(output, prior_outputs, detector_components):
    # Construct entropy of classes
    prior_outputs.iloc[output] += 1
    output_probability = prior_outputs.iloc[:,0]/sum(prior_outputs.values)
    output_entropy = entropy(output_probability)
    detector_components[2] = prior_outputs
    detector_components[5] = output_entropy

def run_detector(detector_components, output, prior_stats, train_out):
    stats = {'pred_c': output, \
			'ed_median': detector_components[3], \
			'mse_mean': detector_components[4], \
			'output_entropy': detector_components[5]}    
    prior_stats = pd.DataFrame(stats, index=[0]) if len(prior_stats) == 0 else prior_stats.append(pd.Series(stats), ignore_index = True)
    prior_stats['mse_cumulative_sum'] = prior_stats.groupby(['pred_c'])['mse_mean'].apply(lambda x: x.shift().expanding().sum())
    prior_stats['ed_cumulative_sum'] = prior_stats.groupby(['pred_c'])['ed_median'].apply(lambda x: x.shift().expanding().sum())
    detector_components[1] = prior_stats
    
    # Get final detector output
    prior_stats['num_query'] = np.arange(1, len(prior_stats) + 1)
    df = prior_stats.merge(train_out, on=['num_query'], how='left')
    df['detector_out'] = alpha * df['mse_cumulative_sum'] + \
                        beta * df['ed_cumulative_sum'] + \
                        gamma * df['output_entropy']
    df['cummax'] = df['detector_out'].cummax()

    df['y_pred'] = 0 #benign
    df.loc[df['detector_out'] > df['cummax']+(df['cummax'] * detector_threshold), 'y_pred'] = 1 #adversary
    df.loc[df['detector_out'] < df['cummax']-(df['cummax'] * detector_threshold), 'y_pred'] = 1 #adversary

    return df['y_pred'].values[-1]

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

    try:
        prior_queries = decrypt_file('./data/prior_queries.pkl')
        prior_stats = decrypt_file('./data/prior_stats.pkl')
        prior_outputs = decrypt_file('./data/prior_outputs.pkl')
    except:
        prior_queries = pd.DataFrame()
        prior_stats = pd.DataFrame()
        prior_outputs = pd.DataFrame.from_dict(dict.fromkeys(range(6),0), orient='index')
    
    train_out = decrypt_file('./data/train_detector.pkl')
    train_out = train_out[train_out['model_type'] == model_type]

    # Load the ONNX model with ONNX Runtime
    ae_ort_session = rt.InferenceSession(decrypt_model(ae_model_path))
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
    X_encoded, X_decoded, output = run_inference(X, ae_ort_session, model_ort_session, model_type)

    # Run detector components using threads for concurrent processing
    detector_components = [0] * 6 # prior_queries, prior_stats, prior_outputs, ed_median, mse_mean, output_entropy
    t1 = threading.Thread(target=run_detector_dist, name='detector_dist', args=(X_encoded, output, prior_queries, detector_components))
    t2 = threading.Thread(target=run_detector_reconstruction, name='detector_rec', args=(X, X_decoded, detector_components)) 
    t3 = threading.Thread(target=run_detector_entropy, name='detector_ent', args=(output, prior_outputs, detector_components))
 
    t1.start()
    t2.start()
    t3.start()
 
    t1.join()
    t2.join()
    t3.join()

    detector_output = run_detector(detector_components, output, prior_stats, train_out)

    # Print output based on detector out
    if detector_output == 1:
        print('Error: You are blocked from using service for 24 hours due to inappropriate usage.')
    else:
        print('Inference result: {}'.format(output))

    # Add runtime and cycles
    total_time = time.time() - start_time
    total_cycles = count_end() - start_cycles
    with open('./data/runtime.txt', 'a') as f:
        f.write('{}, {}\n'.format(total_time, total_cycles))
    
    # Save output file with correct permissions
    encrypt_file(detector_components[0], './data/prior_queries.pkl')
    encrypt_file(detector_components[1], './data/prior_stats.pkl')
    encrypt_file(detector_components[2], './data/prior_outputs.pkl')

if __name__ == '__main__':
    # Get the input file from command-line arguments
    if len(sys.argv) != 2:
        print("Error: The <model_type> command-line arugment is required.")
        sys.exit(1)
    model_type = sys.argv[1]

    # Run the main function
    main(model_type)