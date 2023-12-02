# SODA: Protecting Proprietary Information in On-Device Machine Learning Models

This repository contains the implementation of SODA, a secure on-device application for machine learning model deployment, and experiments discussed in our ACM/IEEE SEC 2023 paper ["SODA: Protecting Proprietary Information in On-Device Machine Learning Models"](https://akanksha-atrey.github.io/papers/atrey2023soda.pdf).

If you use this code or are inspired by our methodology, please cite our SEC paper:

```
@inproceedings{atrey2023soda,
  title={{SODA}: Protecting Proprietary Information in On-Device Machine Learning Models},
  author={Atrey, Akanksha and Sinha, Ritwik and Mitra, Saayan and Shenoy, Prashant},
  booktitle={{ACM/IEEE Symposium on Edge Computing (SEC)}},
  year={2023}
}
```

Please direct all queries to Akanksha Atrey (aatrey at cs dot umass dot edu) or open an issue in this repository.

## About

The growth of low-end hardware has led to a proliferation of machine learning-based services in edge applications. These applications gather contextual information about users and provide some services, such as personalized offers, through a machine learning (ML) model. A growing practice has been to deploy such ML models on the user’s device to reduce latency, maintain user privacy, and minimize continuous reliance on a centralized source. However, deploying ML models on the user’s edge device can leak proprietary information about the service provider. In this work, we investigate on-device ML models that are used to provide mobile services and demonstrate how simple attacks can leak proprietary information of the service provider. We show that different adversaries can easily exploit such models to maximize their profit and accomplish content theft. Motivated by the need to thwart such attacks, we present an end-to-end framework, SODA, for deploying and serving on edge devices while defending against adversarial usage. Our results demonstrate that SODA can detect adversarial usage with 89% accuracy in less than 50 queries with minimal impact on service performance, latency, and storage.

## Setup

### Python

This repository requires Python 3 (>=3.5).

### Packages

All packages used in this repository can be found in the `requirements.txt` file. The following command will install all the packages according to the configuration file:

```
pip install -r requirements.txt
```

## Data

The experiments in this work are executed on two datasets: (1) [UCI Human Activity Recognition](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), and (2) [MNIST Handwritten Digits Classification](http://yann.lecun.com/exdb/mnist/). Please download them into `data/UCI_HAR` and `data/MNIST`, respectively.

## Attacks

This repository contains two types of attacks: (1) exploiting output diversity, and (2) exploiting decision boundaries. The implementation of these attacks can be found in `src/attacks/class_attack.py` and `src/attacks/db_attack.py`, respectively. 

Note, the black box attacks (denoted with a "bb") often take longer to run. It may be worthwhile to run the experiments one at a time. Additionally, swarm scripts are present in the `swarm` folder which may assist further in running the attacks on a slurm-supported server.

### Exploiting Output Diversity

The code for attacking output diversity contains four experiments. To run the class attack that exploits output diversity, execute the following command:

`python3 -m src.attack.class_attack -data_name UCI_HAR -model_type rf -wb_query_attack true -wb_num_feat_attack true -bb_query_attack true -bb_unused_feat_attack true`

### Exploiting Decision Boundaries

The code for attacking decision boundaries contains five experiments. To run the decision boundary attack, execute the following command:

`python3 -m src.attack.db_attack -data_name UCI_HAR -model_type rf -noise_bounds "-0.01 0.01" -exp_num_query true -exp_num_query_bb true -exp_num_query_randfeat true -exp_query_distance true -exp_query_distribution true`

## SODA: Defending On-Device Models

The implementation of SODA can be found in the `src/defense` folder. 

### Training and Executing SODA

The first step is to train an autoencoder model for defending against the attacks. This can be done by executing the following command:

`python3 -m src.defense.defense_training -data_name UCI_HAR -model_type rf`

Following the training of the autoencoder defender, the following command can be executed to run experiments on SODA:

`python3 -m src.defense.defense_detector -data_name UCI_HAR -model_type rf -noise_bounds "-0.01 0.01" -num_queries 100`

### Deploying SODA: A Prototype

A prototype of SODA can be found in the `prototype` folder. This prototype was deployed on a Raspberry Pi.
