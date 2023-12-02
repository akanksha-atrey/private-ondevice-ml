'''
Author: Akanksha Atrey
Description: This script implements Magnet (see https://arxiv.org/pdf/1705.09064.pdf) and focuses on adversarial detection only.
'''

import argparse
import glob
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def magnet(df):
    df['num_query'] = df.groupby(['data', 'model_type', 'user_id']).cumcount()+1
    df = df[df['num_query'] <= 50]
    df = df.groupby(['data', 'model_type', 'num_query'])['mse_mean'].agg('mean').reset_index()
    
    # select max error value based on test set (benign)
    threshold = round(df[df['data'] == 'test']['mse_mean'].max(), 2)
    print('Threshold based on benign set: ', threshold)

    # predict based on mse threshold
    df['y_pred'] = 0 #benign
    df.loc[df['mse_mean'] > threshold, 'y_pred'] = 1 #adversary
    df['y_true'] = 1
    df.loc[df['data']=='train', 'y_true'] = 0
    df.loc[df['data']=='test', 'y_true'] = 0

    print('Train accuracy: ', accuracy_score(df[df['data'] == 'train']['y_true'], df[df['data'] == 'train']['y_pred']))
    print('Benign accuracy: ', accuracy_score(df[df['data'] == 'test']['y_true'], df[df['data'] == 'test']['y_pred']))
    print('Rand accuracy: ', accuracy_score(df[df['data'] == 'rand']['y_true'], df[df['data'] == 'rand']['y_pred']))
    print('Noise accuracy: ', accuracy_score(df[df['data'] == 'noise']['y_true'], df[df['data'] == 'noise']['y_pred']))

    # summarize results
    df_out = df.groupby(['model_type', 'num_query']).apply(lambda x: pd.Series({'precision': precision_score(x['y_true'], x['y_pred']), \
                                                                                'recall': recall_score(x['y_true'], x['y_pred']), \
                                                                                'acc': accuracy_score(x['y_true'], x['y_pred']) })).reset_index()
    df_out[['precision', 'recall', 'acc']] = df_out[['precision', 'recall', 'acc']]*100
    print(df_out[df_out['num_query'] == 50])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_name', type=str, default='UCI_HAR')
    args = parser.parse_args()
    
    # read input files
    dfs = []
    for path in glob.glob('./results/{}/defense/detector*.csv'.format(args.data_name)):
        with open(path) as file:
            data_type = path.split('/')[-1].split('_')[1]
            model_type = path.split('/')[-1].split('_')[2].split('.csv')[0]
            df = pd.read_csv(file)
            df['model_type'] = model_type
            dfs.append(df)
    df_all = pd.concat(dfs).reset_index()
    df_all = df_all.fillna(0)

    magnet(df_all)

if __name__ == '__main__':
    main()