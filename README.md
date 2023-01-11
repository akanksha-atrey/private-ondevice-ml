# Protecting Proprietary Information During On-Device Deployment of Machine Learning Models

This repository contains the implementation of privacy preserving on-device machine learning (ML) models.

Please direct all queries to Akanksha Atrey (aatrey at cs dot umass dot edu) or open an issue in this repository.

## About

Deploying machine learning models on the device reduces latency and helps maintain user privacy. However, serving on the user's device can leak proprietary information about the service provider. In this work, we investigate on-device machine learning models that are used to provide a service and develop a taxonomy of attacks that can leak proprietary information of the service provider. We demonstrate that different adversaries can easily exploit such models to maximize their profit and accomplish content theft. Motivated by the need to thwart such attacks, we present an end-to-end framework that protects the model and defends against adversarial usage.

## Setup

### Python

This repository requires Python 3 (>=3.5).

### Packages

All packages used in this repository can be found in the `requirements.txt` file. The following command will install all the packages according to the configuration file:

```
pip install -r requirements.txt
```