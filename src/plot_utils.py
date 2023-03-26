'''
Author: Akanksha Atrey
Description: This file contains plotting utility functions for different experiments.
'''

from src.models_har import Net

import pandas as pd
import numpy as np
import os
import argparse
import pickle as pkl

import torch
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('text', usetex=False)
plt.rcParams['axes.labelsize']=16

def plot_query_rand_data(df_wb, df_bb, data_name):
    bb_line_styles = ['--', '-.', ':']

    #plot lines
    plt.plot(df_wb[df_wb['model_type'] == 'dt']['query_size'], df_wb[df_wb['model_type'] == 'dt']['accuracy'], label='WB: 0 extra features', color='C0', linestyle='-')
    plt.plot(df_wb[df_wb['model_type'] == 'lr']['query_size'], df_wb[df_wb['model_type'] == 'lr']['accuracy'], color='C1', linestyle='-')
    plt.plot(df_wb[df_wb['model_type'] == 'dnn']['query_size'], df_wb[df_wb['model_type'] == 'dnn']['accuracy'], color='C2', linestyle='-')
    for i,size in enumerate([10,1000,5000]):
        df_bb_size = df_bb[df_bb['num_extra_features']==size]
        plt.plot(df_bb_size[df_bb_size['model_type'] == 'dt']['query_size'], df_bb_size[df_bb_size['model_type'] == 'dt']['accuracy'], label='BB: {} extra features'.format(size), color='C0', linestyle=bb_line_styles[i])
        plt.plot(df_bb_size[df_bb_size['model_type'] == 'lr']['query_size'], df_bb_size[df_bb_size['model_type'] == 'lr']['accuracy'], color='C1', linestyle='--')
        plt.plot(df_bb_size[df_bb_size['model_type'] == 'dnn']['query_size'], df_bb_size[df_bb_size['model_type'] == 'dnn']['accuracy'], color='C2', linestyle='--')
    
    #set axes limits and labels
    plt.xlim([10,5000])
    plt.ylim([50,101])
    plt.xscale('log')
    plt.xlabel("Query Size")
    plt.ylabel("Prediction Space Recovered (%)")
    
    #set legend
    plt.legend(loc='best', title='Feature')
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('black')
    leg.legendHandles[1].set_color('black')
    leg.legendHandles[2].set_color('black')
    leg.legendHandles[3].set_color('black')
    
    plt.savefig('./results/{}/attack/attack_query_size.pdf'.format(data_name), bbox_inches='tight')
    plt.clf()

def plot_query_rand_feat(df, data_name):
    plt.errorbar(df[df['model_type'] == 'dt']['num_features'], df[df['model_type'] == 'dt']['accuracy_mean'], yerr=df[df['model_type'] == 'dt']['accuracy_std'], label='DT', color='C0')
    plt.errorbar(df[df['model_type'] == 'dnn']['num_features'], df[df['model_type'] == 'lr']['accuracy_mean'], yerr=df[df['model_type'] == 'lr']['accuracy_std'], label='LR', color='C1')
    plt.errorbar(df[df['model_type'] == 'dnn']['num_features'], df[df['model_type'] == 'dnn']['accuracy_mean'], yerr=df[df['model_type'] == 'rf']['accuracy_std'], label='DNN', color='C2')
    plt.xlabel("Number of Features Queried")
    plt.ylabel("Prediction Space Recovered (\%)")
    plt.legend(loc='best', title='Model Type')
    plt.savefig('./results/{}/attack/attack_wb_query_featvsaccuracy.pdf'.format(data_name), bbox_inches='tight')
    plt.clf()

def plot_bb_unused_feat(df, data_name):
    bb_line_styles = ['--', '-.', ':']

    #plot lines
    # df = df[df['num_extra_features'] <= 5000]
    df_dnn = df[df['model_type']=='dnn']
    df_dt = df[df['model_type']=='dt']
    df_lr = df[df['model_type']=='lr']
    for i,num_feat in enumerate([10,1000,5000]):
        plt.plot(df_dt[df_dt['query_size'] == num_feat]['num_extra_features'], df_dt[df_dt['query_size'] == num_feat]['runtime'], label=str(num_feat), color='C0', linestyle=bb_line_styles[i])
        plt.plot(df_lr[df_lr['query_size'] == num_feat]['num_extra_features'], df_lr[df_lr['query_size'] == num_feat]['runtime'], color='C1', linestyle=bb_line_styles[i])
        plt.plot(df_dnn[df_dnn['query_size'] == num_feat]['num_extra_features'], df_dnn[df_dnn['query_size'] == num_feat]['runtime'], color='C2', linestyle=bb_line_styles[i])
    
    #set axes limits and labels
    plt.xlabel("Number of Unused Features")
    plt.ylabel("Runtime (s)")
    # plt.xscale('log')

    #set legend
    plt.legend(loc='best', title='Query Size')
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('black')
    leg.legendHandles[1].set_color('black')
    leg.legendHandles[2].set_color('black')

    plt.savefig('./results/{}/attack/attack_bb_query_unusedvsruntime.pdf'.format(data_name), bbox_inches='tight')
    plt.clf()

def plot_db_query_size(df, data_name):
    df = df[df['noise_bounds']=='(-0.01, 0.01)']

    df_dnn = df[df['model_type']=='dnn']
    df_dt = df[df['model_type']=='dt']
    df_lr = df[df['model_type']=='lr']

    colors = ['C0', 'C1', 'C2']
    marker = ['*', 'o', 's', '.', 'v', 'x']

    for i,d in enumerate([df_dt, df_lr, df_dnn]):
        for j,c in enumerate(d['seed_class'].unique()):
            label = c if i==0 else '_nolabel_'
            df_c = d[d['seed_class']==c]
            plt.plot(df_c['num_query'], df_c['same_class_mean'], \
                            c=colors[i], marker=marker[j], label=label)

    plt.xlabel("Query Size")
    plt.ylabel("Number of Exploitations")

    plt.legend(loc='best', title='Seed Class')
    ax = plt.gca()
    leg = ax.get_legend()
    for i,item in enumerate(leg.legendHandles):
        leg.legendHandles[i].set_color('black')

    plt.savefig('./results/{}/attack/attack_db_query_size.pdf'.format(data_name), bbox_inches='tight')
    plt.clf()

def plot_db_noise_range(df, data_name):
    df = df[df['num_query']==100]
    df_grp = df.groupby(['seed_class', 'model_type', 'noise_bounds'])['same_class_mean'].mean().reset_index()
    sns.barplot(data=df_grp, x='model_type', y='same_class_mean', hue='noise_bounds')
    plt.xlabel('Model Type')
    plt.ylabel('Number of Exploitations')
    plt.legend()
    plt.savefig('./results/{}/attack/attack_db_query_noiserange.pdf'.format(data_name), bbox_inches='tight')
    plt.clf()

def plot_db_query_dist(df, data_name):
    df = df[df['model_type'].isin(['rf', 'lr', 'dnn'])]
    id_vars = [col for col in df.columns if col not in ['euc_dist_same', 'euc_dist_diff']]
    df = pd.melt(df, id_vars=id_vars, value_vars=['euc_dist_same', 'euc_dist_diff'], var_name='type', value_name='euc_dist')
    df['model_type'] = df['model_type'].str.upper()

    # violin plot
    sns.violinplot(data=df[df['type']=='euc_dist_same'], x="seed_class", y="euc_dist", hue="model_type")
    
    plt.xlabel('Seed Class')
    plt.ylabel('Distance Between Queries')
    plt.legend(loc='best', title='Model Type')

    plt.savefig('./results/{}/attack/attack_db_query_dist.pdf'.format(data_name), bbox_inches='tight')
    plt.clf()

    # plot histogram of L2 distances of training data
    X_train = pd.read_csv('./data/UCI_HAR/train/X_train.txt', delim_whitespace=True, header=None)

    # plot histogram of L2 distance
    for m in df['model_type'].unique():
        df_m = df[df['model_type'] == m]
        sns.histplot(data=df_m[df_m['type']=='euc_dist_same'], x="euc_dist", hue="model_type", bins=50)
        plt.show()
        plt.clf()

        sns.histplot(data=df_m, x="euc_dist", hue="model_type", bins=50)
        plt.show()
        plt.clf()

def plot_db_query_distribution(df, pca, model, model_type, data_name):
    df = df[df['model_type'] == model_type]
    plt.scatter(df['pca_c1'], df['pca_c2'], c=df['seed_class'], s=0.5)

    # Create a grid of points over the plot
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Apply the decision function of each tree to the grid of points
    x_inverse = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    if model_type == 'dnn':
        _, Z = torch.max(model(torch.from_numpy(x_inverse).float()).data, 1)
        Z = Z.numpy()
    else:
        Z = model.predict(x_inverse)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.6)
    
    plt.scatter(df['pca_c1'], df['pca_c2'], c=df['seed_class'], s=0.5)
    # plt.legend(['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING'])
    plt.title('Exposure of Queries Across Decision Boundaries for {}'.format(model_type.upper()))
    
    plt.savefig('./results/{}/attack/attack_db_query_distr_{}.pdf'.format(data_name, model_type), bbox_inches='tight')
    plt.clf()

def plot_ensemble_accuracies(df, data_name):
    # single rf: 0.9253478113335596 (100 trees)
    # single dt: 0.8496776382762131
    # single lr: 0.9613165931455717
    # single dnn: 0.9446895147607737
    df = df.append({'model_type': 'rf', 'num_models': 1, 'accuracy': 0.8496776382762131}, ignore_index=True)
    df = df.append({'model_type': 'lr', 'num_models': 1, 'accuracy': 0.9613165931455717}, ignore_index=True)
    df = df.append({'model_type': 'dnn', 'num_models': 1, 'accuracy': 0.9446895147607737}, ignore_index=True)

    sns.barplot(data=df, x='num_models', y='accuracy', hue='model_type')
    plt.xlabel("Number of Models in Ensemble")
    plt.ylabel("Accuracy")
    plt.savefig('./results/{}/ensemble/accuracies.pdf'.format(data_name), bbox_inches='tight')
    plt.clf()

def plot_ensemble_attack_rand_query(df, data_name):
    print(df)
    sns.barplot(data=df, x='num_models', y='accuracy', hue='model_type')
    plt.show()
    plt.clf()

def plot_ensemble_attack_rand_feat(df, data_name):
    print(df)
    sns.barplot(data=df, x='num_models', y='accuracy_mean', hue='model_type')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_name', type=str, default='UCI_HAR', help='name of the dataset')
    parser.add_argument('-model_type', type=str, default='rf', help='rf, lr or dnn')
    parser.add_argument('-attack_query_size', type=bool, default=False)
    parser.add_argument('-attack_wb_rand_feat', type=bool, default=False)
    parser.add_argument('-attack_bb_extra', type=bool, default=False)
    parser.add_argument('-attack_db_query_size', type=bool, default=False)
    parser.add_argument('-attack_db_query_dist', type=bool, default=False)
    parser.add_argument('-attack_db_query_distr', type=bool, default=False)
    parser.add_argument('-ensemble_accuracies', type=bool, default=False)
    parser.add_argument('-ensemble_attack_rand_query', type=bool, default=False)
    parser.add_argument('-ensemble_attack_rand_feat', type=bool, default=False)
    args = parser.parse_args()

    #read in results file
    if args.attack_query_size:
        df_results_wb = pd.read_csv('./results/{}/attack/attack_rand_query.csv'.format(args.data_name))
        df_results_bb = pd.read_csv('./results/{}/attack/attack_bb_rand_query.csv'.format(args.data_name))
        plot_query_rand_data(df_results_wb, df_results_bb, args.data_name)

    if args.attack_wb_rand_feat:
        df_results = pd.read_csv('./results/{}/attack/attack_rand_feat.csv'.format(args.data_name))
        plot_query_rand_feat(df_results, args.data_name)

    if args.attack_bb_extra:
        df_results = pd.read_csv('./results/{}/attack/attack_bb_unused_feat.csv'.format(args.data_name))
        plot_bb_unused_feat(df_results, args.data_name)

    if args.attack_db_query_size:
        df_results = pd.read_csv('./results/{}/attack/attack_db_query.csv'.format(args.data_name))
        plot_db_query_size(df_results, args.data_name)
        plot_db_noise_range(df_results, args.data_name)

    if args.attack_db_query_dist:
        df_results = pd.read_csv('./results/{}/attack/attack_db_query_dist.csv'.format(args.data_name))
        plot_db_query_dist(df_results, args.data_name)

    if args.attack_db_query_distr:
        df_results = pd.read_csv('./results/{}/attack/attack_db_query_distribution.csv'.format(args.data_name))
        
        X_train = pd.read_csv('./data/{}/train/X_train.txt'.format(args.data_name), delim_whitespace=True, header=None)

        if args.model_type == 'dt':
            with open('./models/{}/dt.pkl'.format(args.data_name), 'rb') as f:
                model = pkl.load(f)
        elif args.model_type == 'lr':
            with open('./models/{}/lr.pkl'.format(args.data_name), 'rb') as f:
                model = pkl.load(f)
        elif args.model_type == 'dnn':
            model = torch.load('./models/{}/dnn.pt'.format(args.data_name))

        pca = PCA(n_components=2)
        pca.fit(X=X_train)

        plot_db_query_distribution(df_results, pca, model, args.model_type, args.data_name)
    
    if args.ensemble_accuracies:
        dfs = None
        for i in [2,5,10,20]:
            df = pd.read_csv('./results/{}/ensemble/accuracies_ensemble{}.csv'.format(args.data_name, i))
            df = df[df['model_info'] == 'ensemble']
            df['num_models'] = i
            dfs = df if dfs is None else pd.concat([dfs, df])
        plot_ensemble_accuracies(dfs, args.data_name)

    if args.ensemble_attack_rand_query:
        dfs = pd.read_csv('./results/{}/attack/attack_rand_query.csv'.format(args.data_name))
        dfs['model_info'] = 'single'
        dfs['num_models'] = 1
        for i in [2,5,10,20]:
            df = pd.read_csv('./results/{}/attack/attack_ensemble{}_rand_query.csv'.format(args.data_name, i))
            df['model_info'] = 'ensemble'
            df['num_models'] = i
            dfs = pd.concat([dfs, df])
        plot_ensemble_attack_rand_query(dfs, args.data_name)

    if args.ensemble_attack_rand_feat:
        dfs = pd.read_csv('./results/{}/attack/attack_rand_feat.csv'.format(args.data_name))
        dfs['model_info'] = 'single'
        dfs['num_models'] = 1
        for i in [2,5,10,20]:
            df = pd.read_csv('./results/{}/attack/attack_ensemble{}_rand_feat.csv'.format(args.data_name, i))
            df['model_info'] = 'ensemble'
            df['num_models'] = i
            dfs = pd.concat([dfs, df])
        plot_ensemble_attack_rand_feat(dfs, args.data_name)

if __name__ == '__main__':
    main()