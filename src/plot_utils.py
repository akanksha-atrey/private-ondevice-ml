'''
Author: Akanksha Atrey
Description: This file contains plotting utility functions for different experiments.
'''

from src.attack.models_har import Net

import os
import glob
import pandas as pd
import numpy as np
import argparse
import pickle as pkl

import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import seaborn as sns
plt.rc('text', usetex=False)
plt.rcParams['axes.labelsize']=16

def plot_query_rand_data(df_wb, df_bb, data_name):
    bb_line_styles = ['--', '-.', ':']

    #plot lines
    plt.plot(df_wb[df_wb['model_type'] == 'rf']['query_size'], df_wb[df_wb['model_type'] == 'rf']['accuracy'], label='WB: 0 extra features', color='C0', linestyle='-')
    plt.plot(df_wb[df_wb['model_type'] == 'lr']['query_size'], df_wb[df_wb['model_type'] == 'lr']['accuracy'], color='C1', linestyle='-')
    plt.plot(df_wb[df_wb['model_type'] == 'dnn']['query_size'], df_wb[df_wb['model_type'] == 'dnn']['accuracy'], color='C2', linestyle='-')
    for i,size in enumerate([10,1000,5000]):
        df_bb_size = df_bb[df_bb['num_extra_features']==size]
        plt.plot(df_bb_size[df_bb_size['model_type'] == 'rf']['query_size'], df_bb_size[df_bb_size['model_type'] == 'rf']['accuracy'], label='BB: {} extra features'.format(size), color='C0', linestyle=bb_line_styles[i])
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
    plt.errorbar(df[df['model_type'] == 'rf']['num_features'], df[df['model_type'] == 'rf']['accuracy_mean'], yerr=df[df['model_type'] == 'rf']['accuracy_std'], label='RF', color='C0')
    plt.errorbar(df[df['model_type'] == 'lr']['num_features'], df[df['model_type'] == 'lr']['accuracy_mean'], yerr=df[df['model_type'] == 'lr']['accuracy_std'], label='LR', color='C1')
    plt.errorbar(df[df['model_type'] == 'dnn']['num_features'], df[df['model_type'] == 'dnn']['accuracy_mean'], yerr=df[df['model_type'] == 'dnn']['accuracy_std'], label='DNN', color='C2')
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
    df_rf = df[df['model_type']=='rf']
    df_lr = df[df['model_type']=='lr']
    for i,num_feat in enumerate([10,1000,5000]):
        plt.plot(df_rf[df_rf['query_size'] == num_feat]['num_extra_features'], df_rf[df_rf['query_size'] == num_feat]['runtime'], label=str(num_feat), color='C0', linestyle=bb_line_styles[i])
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
    df_rf = df[df['model_type']=='rf']
    df_lr = df[df['model_type']=='lr']

    colors = ['C0', 'C1', 'C2']
    marker = ['*', 'o', 's', '.', 'v', 'x']

    for i,d in enumerate([df_rf, df_lr, df_dnn]):
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
    palette = {"rf": "C0", "lr": "C1", "dnn": "C2"}
    df = df[(df['num_query']==100) & (df['model_type'].isin(['rf', 'lr', 'dnn']))]
    df_grp = df.groupby(['model_type', 'noise_bounds'])['same_class_mean'].mean().reset_index()
    ax = sns.barplot(data=df_grp, x='noise_bounds', y='same_class_mean', hue='model_type', palette=palette)
    ax.legend_.remove()
    plt.xlabel('Noise Bounds')
    plt.ylabel('Number of Exploitations')
    plt.savefig('./results/{}/attack/attack_db_query_noiserange.pdf'.format(data_name), bbox_inches='tight')
    plt.clf()

def plot_db_query_dist(df, data_name):
    palette = {"RF": "C0", "LR": "C1", "DNN": "C2"}
    df = df[df['noise_bounds']=='(-0.01, 0.01)']
    df = df[df['model_type'].isin(['rf', 'lr', 'dnn'])]
    id_vars = [col for col in df.columns if col not in ['euc_dist_same', 'euc_dist_diff']]
    df = pd.melt(df, id_vars=id_vars, value_vars=['euc_dist_same', 'euc_dist_diff'], var_name='type', value_name='euc_dist')
    df['model_type'] = df['model_type'].str.upper()

    # violin plot
    sns.violinplot(data=df[df['type']=='euc_dist_same'], x="seed_class", y="euc_dist", hue="model_type", palette=palette)
    
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

def plot_db_runtime(df_wb, df_bb, data_name):
    bb_line_styles = ['-', '--', '-.', ':']

    #plot runtime
    df_wb_dnn = df_wb[(df_wb['noise_bounds'] == '(-0.01, 0.01)') & (df_bb['model_type']=='dnn')]
    df_wb_rf = df_wb[(df_wb['noise_bounds'] == '(-0.01, 0.01)') & (df_bb['model_type']=='rf')]
    df_wb_lr = df_wb[(df_wb['noise_bounds'] == '(-0.01, 0.01)') & (df_bb['model_type']=='lr')]

    df_bb_dnn = df_bb[df_bb['model_type']=='dnn']
    df_bb_rf = df_bb[df_bb['model_type']=='rf']
    df_bb_lr = df_bb[df_bb['model_type']=='lr']
    for i,num_feat in enumerate([10,1000,5000]):
        # plt.plot(df_wb[df_wb['query_size'] == num_feat]['num_extra_features'], df_wb[df_wb['query_size'] == num_feat]['runtime'], label=str(num_feat), color='C0', linestyle=bb_line_styles[i])
        plt.plot(df_bb_rf[df_bb_rf['num_query'] == num_feat]['num_extra_features'], df_bb_rf[df_bb_rf['num_query'] == num_feat]['runtime'], label=str(num_feat), color='C0', linestyle=bb_line_styles[i])
        plt.plot(df_bb_lr[df_bb_lr['num_query'] == num_feat]['num_extra_features'], df_bb_lr[df_bb_lr['num_query'] == num_feat]['runtime'], color='C1', linestyle=bb_line_styles[i])
        plt.plot(df_bb_dnn[df_bb_dnn['num_query'] == num_feat]['num_extra_features'], df_bb_dnn[df_bb_dnn['num_query'] == num_feat]['runtime'], color='C2', linestyle=bb_line_styles[i])
    
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

    plt.savefig('./results/{}/attack/attack_db_bb_unusedvsruntime.pdf'.format(data_name), bbox_inches='tight')
    plt.clf()

def plot_db_rand_feat(df, data_name):
    sns.lineplot(data=df[df['model_type'] == 'rf'], x='num_features', y='same_class_mean', style='seed_class', color='C0', label='_nolegend_')
    sns.lineplot(data=df[df['model_type'] == 'lr'], x='num_features', y='same_class_mean', style='seed_class', color='C1', label='_nolegend_')
    ax = sns.lineplot(data=df[df['model_type'] == 'dnn'], x='num_features', y='same_class_mean', style='seed_class', color='C2', label='_nolegend_')
    
    handles, labels = ax.get_legend_handles_labels()
    labels = labels[:6]

    plt.xlabel("Number of Features Perturbed")
    plt.ylabel("Percentage of Exploitatons (%)")
    plt.legend(loc='best', title='Seed Class', handles=handles, labels=labels)
    plt.savefig('./results/{}/attack/attack_db_randfeatvsexploitation.pdf'.format(data_name), bbox_inches='tight')
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

def plot_defense(df, data_name, alpha=0.33, beta=0.33, gamma=0.33, detector_threshold=0.2):
    df['detector_out'] = alpha*df['mse_cumulative_sum'] + beta*df['ed_enc_cumulative_sum'] + gamma*df['output_entropy']
    df['num_query'] = df.groupby(['data', 'model_type', 'user_id']).cumcount()+1
    df = df[df['num_query'] <= 50]
    detector_sum = df.groupby(['data', 'model_type', 'num_query'])['detector_out'].agg('mean').reset_index()
    print(detector_sum)

    ################# LEAKAGE RATE RESULTS #################
    # plot detector output against queries
    fig, ax = plt.subplots()
    sns.lineplot(data=detector_sum, x='num_query', y='detector_out', hue='data', style='model_type', palette='husl', hue_order=['train', 'test', 'rand', 'noise'])    
    plt.xlabel('Number of Queries')
    plt.ylabel('Leakage Rate')
    plt.yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    print(handles, labels)
    labels = ['Input Type', 'Train', 'Benign', 'Adv Random', 'Adv Perturbation', 'Model Type', 'DNN', 'LR', 'RF']
    order = [0, 1, 2, 3, 4, 5, 8, 7, 6]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.savefig('./results/{}/defense/leakage_query.pdf'.format(data_name), bbox_inches='tight')
    plt.clf()

    # plot each individual component of final detector
    df_melted = pd.melt(df, id_vars=['data', 'model_type', 'num_query', 'user_id'], \
            value_vars=['ed_enc_cumulative_sum', 'mse_cumulative_sum', 'output_entropy'], \
            var_name='component', value_name='value')
    df_melted = df_melted[df_melted['num_query'] == 50]
    df_melted = df_melted.groupby(['data', 'model_type', 'component'], as_index=False)['value'].agg('mean')
    df_melted['data_model'] = df_melted['data'] + '_' + df_melted['model_type']
    bar_orders = ['train_rf', 'train_lr', 'train_dnn', 'test_rf', 'test_lr', 'test_dnn', \
                  'rand_rf', 'rand_lr', 'rand_dnn', 'noise_rf', 'noise_lr', 'noise_dnn']
    
    fig, ax = plt.subplots()
    sns.barplot(data = df_melted, x = 'data_model', y = 'value', hue = 'component', palette = 'Set2', \
                order = bar_orders)
    plt.yscale('log')
    plt.xlabel('')
    plt.ylabel('Component Value')
    ax.set_xticklabels(['RF', 'LR', 'DNN']*4) #first layer x labels
    ax2 = ax.twiny() #second layer x labels
    ax2.spines["bottom"].set_position(("axes", -0.10))
    ax2.tick_params('both', length=0, width=0, which='minor')
    ax2.tick_params('both', direction='in', which='major')
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")

    ax2.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(ticker.FixedLocator([0.125, 0.375, 0.625, 0.875]))
    ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(['Train', 'Benign', 'Adv Random', 'Adv Perturbation']))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles = handles, labels = ['Query Distance', 'Reconstruction Error', 'Output Entropy'], title = 'Component')
    plt.savefig('./results/{}/defense/leakage_component.pdf'.format(data_name), bbox_inches='tight')
    plt.clf()

    # for model_type in df_melted['model_type'].unique():
    #     df_melted_model = df_melted[df_melted['model_type'] == model_type]
    #     sns.barplot(data=df_melted_model, x='data', y='value', hue='component')
    #     plt.xlabel('Data Type')
    #     plt.ylabel('Value')
    #     plt.title('Component Values for {}'.format(model_type))
    #     # plt.savefig('./results/{}/defense/defense_component_scaled_{}.pdf'.format(data_name, model_type), bbox_inches='tight')
    #     # plt.clf()

    # plot detector out with ranging leakage component thresholds
    leakage_thresholds = [[0.3, 0.3, 0.3], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    df_leakage_thresholds = None
    for a, b, g in leakage_thresholds:
        df_leakage = df[df['num_query'] == 50]
        df_leakage['detector_out'] = a*df_leakage['mse_cumulative_sum'] + b*df_leakage['ed_enc_cumulative_sum'] + g*df_leakage['output_entropy']
        df_leakage = df_leakage[df_leakage['num_query'] == 50]
        det_sum = df_leakage.groupby(['data', 'model_type', 'num_query'])['detector_out'].agg('mean').reset_index()
        det_sum['threshold'] = '[{}, {}, {}]'.format(a, b, g)
        df_leakage_thresholds = det_sum if df_leakage_thresholds is None else pd.concat([df_leakage_thresholds, det_sum])

    fig, ax = plt.subplots()
    sns.barplot(data = df_leakage_thresholds, x = 'threshold', y = 'detector_out', hue = 'data', \
                hue_order = ['train', 'test', 'rand', 'noise'], palette='husl')
    plt.yscale('log')
    plt.xlabel('Leakage Component Dependency')
    plt.ylabel('Leakage Rate')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles = handles, labels = ['Train', 'Benign', 'Adv Random', 'Adv Perturbation'], title = 'Input Type')
    ax.set_xticklabels(['Equal', 'Dist-Rec', 'Dist-Ent', 'Rec-Ent'])
    plt.savefig('./results/{}/defense/leakage_thresholds.pdf'.format(data_name), bbox_inches='tight')
    plt.clf()

    ################# DETECTOR OUTPUT RESULTS #################
    # plot detector accuracy with cummax at detector_threshold
    detector_sum['cummax'] = detector_sum.groupby(['data', 'model_type'])['detector_out'].cummax()
    df = df.merge(detector_sum[detector_sum['data']=='train'][['model_type', 'num_query', 'cummax']], on=['num_query', 'model_type'], how='left')
    
    df['y_pred'] = 0 #benign
    df.loc[df['detector_out'] > df['cummax']+(df['cummax']*detector_threshold), 'y_pred'] = 1 #adversary
    df.loc[df['detector_out'] < df['cummax']-(df['cummax']*detector_threshold), 'y_pred'] = 1 #adversary
    df['y_true'] = 1
    df.loc[df['data']=='train', 'y_true'] = 0
    df.loc[df['data']=='test', 'y_true'] = 0

    # df['y_rand'] = np.random.choice([0, 1], size=len(df))

    df_out = df.groupby(['model_type', 'num_query']).apply(lambda x: pd.Series({'precision': precision_score(x['y_true'], x['y_pred']), \
                                                                                'recall': recall_score(x['y_true'], x['y_pred']), \
                                                                                'acc': accuracy_score(x['y_true'], x['y_pred']) })).reset_index()
    df_out[['precision', 'recall', 'acc']] = df_out[['precision', 'recall', 'acc']]*100
    print(df_out)
    fig, ax = plt.subplots()
    # sns.lineplot(data=df_out, x='num_query', y='f1_score', hue='model_type')#, linestyle='-', label='Accuracy')
    sns.lineplot(data=df_out, x='num_query', y='precision', hue='model_type', linestyle='-.', label='Precision', hue_order = ['rf', 'lr', 'dnn'])
    sns.lineplot(data=df_out, x='num_query', y='recall', hue='model_type', linestyle='-', label='Recall', hue_order = ['rf', 'lr', 'dnn'])
    plt.xlabel('Number of Queries')
    plt.ylabel('Detector Performance (%)')    
    custom_lines = [Line2D([], [], color="none", label='Model Type'),
    				Line2D([0], [0], color='C0', linestyle='-', label='RF'),
    				Line2D([0], [0], color='C1', linestyle='-', label='LR'),
    				Line2D([0], [0], color='C2', linestyle='-', label='DNN'),
    				Line2D([], [], color="none", label='Measure'),
    				Line2D([0], [0], color='black', linestyle='-.', label='Precision'),
                    Line2D([0], [0], color='black', linestyle='-', label='Recall')]
    ax.legend(handles=custom_lines, labels=['Model Type', 'RF', 'LR', 'DNN', 'Measure', 'Precision', 'Recall'])
    plt.savefig('./results/{}/defense/detector_performance.pdf'.format(data_name), bbox_inches='tight')
    plt.clf()

    # plot detector accuracy for 50 queries with differing thresholds
    df_multi_threshold = None
    for t in [0.1, 0.2, 0.3, 0.4]:
        df['y_pred'] = 0 #benign
        df.loc[df['detector_out'] > df['cummax']+(df['cummax'] * t), 'y_pred'] = 1 #adversary
        df.loc[df['detector_out'] < df['cummax']-(df['cummax'] * t), 'y_pred'] = 1 #adversary
        df['y_true'] = 1
        df.loc[df['data']=='train', 'y_true'] = 0
        df.loc[df['data']=='test', 'y_true'] = 0

        df_out = df.groupby(['model_type', 'num_query']).apply(lambda x: pd.Series({'precision': precision_score(x['y_true'], x['y_pred']), \
                                                                                'recall': recall_score(x['y_true'], x['y_pred']), \
                                                                                'f1_score': f1_score(x['y_true'], x['y_pred']) })).reset_index()

        df_out[['precision', 'recall', 'f1_score']] = df_out[['precision', 'recall', 'f1_score']]*100
        df_out['threshold'] = t
        df_multi_threshold = df_out if df_multi_threshold is None else pd.concat([df_multi_threshold, df_out])
        
    df_multi_threshold = df_multi_threshold[df_multi_threshold['num_query'] == 50].reset_index()
    fig, ax = plt.subplots()
    sns.lineplot(data = df_multi_threshold, x = 'threshold', y = 'precision', hue = 'model_type', linestyle='-.', hue_order = ['rf', 'lr', 'dnn'])
    sns.lineplot(data = df_multi_threshold, x = 'threshold', y = 'recall', hue = 'model_type', linestyle='-', hue_order = ['rf', 'lr', 'dnn'])
    plt.xlabel('Detector Threshold')
    plt.ylabel('Detector Performance (%)')
    custom_lines = [Line2D([], [], color="none", label='Model Type'),
    				Line2D([0], [0], color='C0', linestyle='-', label='RF'),
    				Line2D([0], [0], color='C1', linestyle='-', label='LR'),
    				Line2D([0], [0], color='C2', linestyle='-', label='DNN'),
    				Line2D([], [], color="none", label='Measure'),
    				Line2D([0], [0], color='black', linestyle='-.', label='Precision'),
                    Line2D([0], [0], color='black', linestyle='-', label='Recall')]
    ax.legend(handles=custom_lines, labels=['Model Type', 'RF', 'LR', 'DNN', 'Measure', 'Precision', 'Recall'])
    plt.savefig('./results/{}/defense/detector_thresholds.pdf'.format(data_name), bbox_inches='tight')
    plt.clf()

    # plot queries by runtime
    min_time = df[['ed_time', 'mse_time', 'entropy_time']].min().min()
    max_time = df[['ed_time', 'mse_time', 'entropy_time']].max().max()
    print(min_time, max_time)
    print(df['ed_time'].subtract(min_time))
    df['ed_time'] = df['ed_time'].subtract(min_time).divide(max_time - min_time)
    df['mse_time'] = df['mse_time'].subtract(min_time).divide(max_time - min_time)
    df['entropy_time'] = df['entropy_time'].subtract(min_time).divide(max_time - min_time)
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x='num_query', y='ed_time', label='Distance', hue='model_type', hue_order = ['rf', 'lr', 'dnn'], linestyle = '-')
    sns.lineplot(data=df, x='num_query', y='mse_time', label='MSE', hue='model_type', hue_order = ['rf', 'lr', 'dnn'], linestyle = '--')
    sns.lineplot(data=df, x='num_query', y='entropy_time', label='Entropy', hue='model_type', hue_order = ['rf', 'lr', 'dnn'], linestyle = '-.')
    plt.legend(title='Detector Component')
    custom_lines = [Line2D([], [], color="none", label='Model Type'),
    				Line2D([0], [0], color='C0', linestyle='-', label='RF'),
    				Line2D([0], [0], color='C1', linestyle='-', label='LR'),
    				Line2D([0], [0], color='C2', linestyle='-', label='DNN'),
    				Line2D([], [], color="none", label='Component'),
    				Line2D([0], [0], color='black', linestyle='-', label='Distance'),
                    Line2D([0], [0], color='black', linestyle='--', label='Reconstruction'),
                    Line2D([0], [0], color='black', linestyle='-.', label='Entropy')]
    ax.legend(handles=custom_lines, labels=['Model Type', 'RF', 'LR', 'DNN', 'Component', 'Query Distance', 'Reconstruction Error', 'Output Entropy'])
    plt.xlabel('Number of Queries')
    plt.ylabel('Normalized Runtime (s)')
    # plt.yscale('log')
    plt.savefig('./results/{}/defense/detector_runtime.pdf'.format(data_name), bbox_inches='tight')
    plt.clf()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_name', type=str, default='UCI_HAR', help='name of the dataset')
    parser.add_argument('-model_type', type=str, default='rf', help='rf, lr or dnn')
    parser.add_argument('-attack_query_size', type=bool, default=False)
    parser.add_argument('-attack_wb_rand_feat', type=bool, default=False)
    parser.add_argument('-attack_bb_extra', type=bool, default=False)
    parser.add_argument('-attack_db_query_size', type=bool, default=False)
    parser.add_argument('-attack_db_runtime', type=bool, default=False)
    parser.add_argument('-attack_db_rand_feat', type=bool, default=False)
    parser.add_argument('-attack_db_query_dist', type=bool, default=False)
    parser.add_argument('-attack_db_query_distr', type=bool, default=False)
    parser.add_argument('-ensemble_accuracies', type=bool, default=False)
    parser.add_argument('-ensemble_attack_rand_query', type=bool, default=False)
    parser.add_argument('-ensemble_attack_rand_feat', type=bool, default=False)
    parser.add_argument('-defense', type=bool, default=False)
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

    if args.attack_db_runtime:
        df_wb_results = pd.read_csv('./results/{}/attack/attack_db_query.csv'.format(args.data_name))
        df_bb_results = pd.read_csv('./results/{}/attack/attack_db_query_bb.csv'.format(args.data_name))
        plot_db_runtime(df_wb_results, df_bb_results, args.data_name)

    if args.attack_db_rand_feat:
        df_results = pd.read_csv('./results/{}/attack/attack_db_query_randfeat.csv'.format(args.data_name))
        plot_db_rand_feat(df_results, args.data_name)

    if args.attack_db_query_dist:
        df_results = pd.read_csv('./results/{}/attack/attack_db_query_dist.csv'.format(args.data_name))
        plot_db_query_dist(df_results, args.data_name)

    if args.attack_db_query_distr:
        df_results = pd.read_csv('./results/{}/attack/attack_db_query_distribution.csv'.format(args.data_name))
        
        X_train = pd.read_csv('./data/{}/train/X_train.txt'.format(args.data_name), delim_whitespace=True, header=None)

        if args.model_type == 'rf':
            with open('./models/{}/rf.pkl'.format(args.data_name), 'rb') as f:
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

    if args.defense:
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

        # scale all values by training data
        scaler = MinMaxScaler()
        scaler.fit(df_all[df_all['data'] == 'train'][['mse_cumulative_sum', 'ed_enc_cumulative_sum', 'output_entropy']])
        df_all[['mse_cumulative_sum', 'ed_enc_cumulative_sum', 'output_entropy']] = scaler.transform(df_all[['mse_cumulative_sum', 'ed_enc_cumulative_sum', 'output_entropy']])

        # plot results
        print('now plotting')
        plot_defense(df_all, args.data_name, alpha=0.33, beta=0.33, gamma=0.33)

if __name__ == '__main__':
    main()