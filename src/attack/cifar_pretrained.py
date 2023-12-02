import numpy as np
import pandas as pd

from huggingface_hub import login
import transformers
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification

import torch

from datasets import load_dataset

def class_attack(X, model, num_queries, feat_min, feat_max):
    X_augment = np.repeat(X, repeats = num_queries, axis = 0)
    X_augment_indices = np.repeat(range(len(X)), repeats = num_queries)
    X_augment_noise = np.random.uniform(feat_min, feat_max, size=len(X_augment)*3*224*224).reshape(len(X_augment),3,224,224)
    outputs = model(torch.from_numpy(X_augment_noise).float())
    logits = outputs.logits
    result_attack = logits.argmax(-1)

    df_pred = pd.DataFrame.from_dict({'index': X_augment_indices, 'pred': result_attack})
    attack_acc = df_pred['pred'].groupby(df_pred.index).nunique().mean()/100*100 # num_classes * 100 for percentage
    print('Class attack accuracy: ', attack_acc)

def main():
    DEBUG = False

    # Your Hugging Face username and token
    api_token = "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"
    login(token = api_token)

    # Load data
    ds = load_dataset("cifar100", split='test', streaming = True)#, cache_dir='/mnt/nfs/scratch1/aatrey')
    ds_shuffled = ds.shuffle(seed=40)
    ds_rand = ds_shuffled.take(100) #100 random data points
    
    # Load model
    feature_extractor = AutoFeatureExtractor.from_pretrained("MazenAmria/swin-tiny-finetuned-cifar100")
    model = AutoModelForImageClassification.from_pretrained("MazenAmria/swin-tiny-finetuned-cifar100")

    # Process data
    X = []
    for i, img in enumerate(ds_rand):
        try:
            inputs = feature_extractor(images=img['img'], return_tensors="pt")

            X.append(np.array(inputs['pixel_values']))

            if DEBUG:
                outputs = model(**inputs)
                logits = outputs.logits

                # model predicts one of the 100 CIFAR classes
                predicted_class_idx = logits.argmax(-1).item()
                print(i, img['fine_label'], predicted_class_idx)
        except:
            print("An exception occurred")

    X = np.array(X)
    print('Number of data points: ', X.shape)
    # np.save('./data/CIFAR/test.npy', X)

    # Run class attack
    # X = np.load('./data/CIFAR/test.npy')
    feat_min = X.min()
    feat_max = X.max()
    class_attack(X, model, 100, feat_min, feat_max)

if __name__ == '__main__':
	main()