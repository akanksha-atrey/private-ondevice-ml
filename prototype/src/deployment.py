import sys
import time
from hwcounter import count, count_end

import onnxruntime as rt
import numpy as np
import pandas as pd
import pickle as pkl

from scipy.spatial.distance import cdist
from scipy.stats import entropy

# Thresholds
alpha = 0.33
beta = 0.33
gamma = 0.33
detector_threshold = 0.2

# Path to the bundled ONNX model file
ae_model_path = './models/UCI_HAR/prototype/ae_quantized.onnx'
rf_model_path = './models/UCI_HAR/prototype/rf.onnx'
lr_model_path = './models/UCI_HAR/prototype/lr.onnx'
dnn_model_path = './models/UCI_HAR/prototype/dnn_quantized.onnx'

def run_inference(X, ae_rt_session, model_rt_session, model_type):
    # Perform inference using the loaded ONNX model
    input_name = ae_rt_session.get_inputs()[0].name
    X_decoded, X_encoded = ae_rt_session.run(None, {input_name: X.astype(np.float32)})
    
    input_name = model_rt_session.get_inputs()[0].name
    output = model_rt_session.run(None, {input_name: X_encoded.astype(np.float32)})[0]

    if model_type == 'dnn':
        output = output.argmax(axis = 1)

    return X_encoded, X_decoded, output[0]

def run_detector(X, X_encoded, X_decoded, output, prior_queries, prior_stats, prior_outputs, train_out):
    
    # Measure difference in input to closest element in class c
    ed_median = 0
    if len(prior_queries) != 0:
        prior_queries_same_class = prior_queries[prior_queries['pred_c'] == output]

        if len(prior_queries_same_class) != 0:
             distances = cdist(X_encoded.reshape(1,-1), prior_queries_same_class.drop(columns=['pred_c']), metric='euclidean')
             ed_median = np.median(distances, axis=1)[0]
    
    # Measure reconstruction error (x vs ae(x)) to separate out anomalies/OOD queries
    mse = (X - X_decoded)**2

    # Construct entropy of classes
    prior_outputs[output] += 1
    output_probability = [v/sum(prior_outputs.values()) for k,v in prior_outputs.items()]
    output_entropy = entropy(output_probability)

    # Save output
    encoding_df = pd.DataFrame(X_encoded, columns = ['Column_' + str(i) for i in range(X_encoded.shape[1])])
    encoding_df['pred_c'] = output
    prior_queries = encoding_df if len(prior_queries) == 0 is None else prior_queries.append(encoding_df, ignore_index = True)

    stats = {'pred_c': output, \
			'ed_median': ed_median, \
			'mse_mean': np.mean(mse), \
			'output_entropy': output_entropy}    
    prior_stats = pd.DataFrame(stats, index=[0]) if len(prior_stats) == 0 is None else prior_stats.append(pd.Series(stats), ignore_index = True)
    prior_stats['mse_cumulative_sum'] = prior_stats.groupby(['pred_c'])['mse_mean'].apply(lambda x: x.shift().expanding().sum())
    prior_stats['ed_cumulative_sum'] = prior_stats.groupby(['pred_c'])['ed_median'].apply(lambda x: x.shift().expanding().sum())
    
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

    return df['y_pred'].values[-1], prior_queries, prior_stats, prior_outputs

def main(model_type):
    start_cycles = count()
    start_time = time.time()

    # Read input and other resource files
    X = np.genfromtxt('./prototype/data/input.txt').reshape(1, -1)

    try:
        prior_queries = pd.read_csv('./prototype/data/prior_queries.csv')
        prior_stats = pd.read_csv('./prototype/data/prior_stats.csv')
        with open('./prototype/data/prior_outputs.pkl', 'rb') as f:
            prior_outputs = pkl.load(f)
    except:
        prior_queries = pd.DataFrame()
        prior_stats = pd.DataFrame()
        prior_outputs = dict.fromkeys(range(6),0)
    
    train_out = pd.read_csv('./prototype/data/train_detector.csv')
    train_out = train_out[train_out['model_type'] == model_type]

    # Load the ONNX model with ONNX Runtime
    ae_ort_session = rt.InferenceSession(ae_model_path)
    if model_type == 'rf':
        model_ort_session = rt.InferenceSession(rf_model_path)
    elif model_type == 'lr':
        model_ort_session = rt.InferenceSession(lr_model_path)
    elif model_type == 'dnn':
        model_ort_session = rt.InferenceSession(dnn_model_path)
    else:
        print("Error: Model type is not RF, LR, or DNN.")
        sys.exit(1)

    # Run inference
    X_encoded, X_decoded, output = run_inference(X, ae_ort_session, model_ort_session, model_type)

    # Run detector
    detector_output, prior_queries, prior_stats, prior_outputs = run_detector(X, X_encoded, X_decoded, output, prior_queries, prior_stats, prior_outputs, train_out)
    prior_queries.to_csv('./prototype/data/prior_queries.csv', index=False)
    prior_stats.to_csv('./prototype/data/prior_stats.csv', index=False)
    print(detector_output)
    with open('./prototype/data/prior_outputs.pkl', 'wb') as f:
        pkl.dump(prior_outputs, f)

    # Print output based on detector out
    if detector_output == 1:
        print('Error: You are blocked from using service for 24 hours due to inappropriate usage.')
    else:
        print('Inference result: {}'.format(output))

    # Add runtime and cycles
    total_time = time.time() - start_time
    total_cycles = count_end() - start_cycles
    with open('./prototype/data/runtime.txt', 'a') as f:
        f.write('{}, {}\n'.format(total_time, total_cycles))

if __name__ == '__main__':
    # Get the input file from command-line arguments
    if len(sys.argv) != 2:
        print("Error: The <model_type> command-line arugment is required.")
        sys.exit(1)
    model_type = sys.argv[1]

    # Run the main function
    main(model_type)