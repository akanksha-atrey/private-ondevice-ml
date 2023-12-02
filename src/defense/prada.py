'''
Author: Akanksha Atrey
Description: This script implements PRADA (see https://arxiv.org/pdf/1805.02628.pdf).
'''

import argparse
import glob
import pandas as pd
from scipy.stats import shapiro
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def prada(df, threshold=0.95):
    df['num_query'] = df.groupby(['data', 'model_type', 'user_id']).cumcount()+1
    df = df[df['num_query'] <= 50]
    df = df.groupby(['data', 'model_type', 'user_id']).apply(lambda x: shapiro(x['ed_raw_min'])[0]).reset_index(name='shapiro_test')

    # predict based on euclidean distance threshold
    df['y_pred'] = 0 #benign
    df.loc[df['shapiro_test'] < threshold, 'y_pred'] = 1 #adversary
    df['y_true'] = 1
    df.loc[df['data']=='train', 'y_true'] = 0
    df.loc[df['data']=='test', 'y_true'] = 0

    print('Train accuracy: ', accuracy_score(df[df['data'] == 'train']['y_true'], df[df['data'] == 'train']['y_pred']))
    print('Benign accuracy: ', accuracy_score(df[df['data'] == 'test']['y_true'], df[df['data'] == 'test']['y_pred']))
    print('Rand accuracy: ', accuracy_score(df[df['data'] == 'rand']['y_true'], df[df['data'] == 'rand']['y_pred']))
    print('Noise accuracy: ', accuracy_score(df[df['data'] == 'noise']['y_true'], df[df['data'] == 'noise']['y_pred']))

    # summarize results
    df_out = df.groupby(['model_type']).apply(lambda x: pd.Series({'precision': precision_score(x['y_true'], x['y_pred']), \
                                                                                'recall': recall_score(x['y_true'], x['y_pred']), \
                                                                                'acc': accuracy_score(x['y_true'], x['y_pred']) })).reset_index()
    df_out[['precision', 'recall', 'acc']] = df_out[['precision', 'recall', 'acc']]*100
    print(df_out)

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

    prada(df_all)

if __name__ == '__main__':
    main()