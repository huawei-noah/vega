# MF-ASC: Multi-Fidelity neural Architecture Search with Co-kriging

MF-ASC is an individual optimization method (no weight sharing), that works with arbitrary search spaces.
The search algorithm is described in the [paper](https://dl.acm.org/doi/10.1145/3292500.3330893) (see algorithm 1).

## 1. Algorithm Introduction

Multifidelity optimization is applied in the design of complex systems, where a computationally expensive high-fidelity objective function is approximated by a less expensive low-fidelity function and a few high-fidelity samples. In the context of neural architecture search, high- and low-fidelity evaluations are defined by the number of training steps before measuring networks' quality metrics on a validation dataset. MF-ASC algorithm addresses the exploration-exploitation dilemma using the acquisition criterion called upper confidence bound (UCB); we provide the means to integrate high- and low-fidelity sources by Bayesian multifidelity inference with co-kriging schema. 

## 2. Algorithm Principles

At each step the generator decides which fidelity to use and samples an item from the dataset that has not been evaluated with this fidelity level so far. A parameter r ≥ 0 determines the ratio of low-fidelity to high-fidelity calls; this parameter should be provided by the user.
For each fidelity choice, the algorithm first computes the parameters of a regression model using the current state information; it returns the item that maximizes the UCB acquisition criterion by the regression model for either fidelity, excluding previously chosen items. The parameter beta defines exploration and exploitation trade-off: large values favor exploring items having high uncertainty in quality, while small values favor exploiting items having high expeted quality.

## 3. Search Space

The algorithm is applicable to any search space that can be encoded into ℝ^n.

We use a [MobileNetV2 search space](https://arxiv.org/abs/1906.09607) to demonstrate the work of MF-ASC.
The search space includes various combinations of values of repetitions and channels for each layer of the MobileNetV2.

## 4. Usage Guide

For details about how to search a model, see the following configuration file for parameter setting:

- vega/examples/nas/mfasc/mfasc.yml

The configuration of the search algorithm includes the following parameters:

| parameter | desc |
| :--: | :-- |
| batch_size | the number randomly sampled candidates to be assessed by the search method at each iteration of sampling; the best candidate according to UCB criterion is sampled |
| prior_rho | prior correlation between low- and high- fidelity quality metrics |
| beta | parameter beta for the algorithm |
| max_budget | the maximum number of training epochs in total for low- and high-fidelity evaluations |
| hf_epochs | the number of training epochs for high-fidelity evaluation |
| lf_epochs | the number of training epochs for low-fidelity evaluation |
| fidelity_ratio | parameter r for the algorithm |
| min_hf_sample_size | the minimum amount of high-fidelity evaluations (sampled randomly prior to the active search process) |
| min_lf_sample_size | the minimum amount of low-fidelity evaluations (sampled randomly prior to the active search process) |
| predictor_type | either 'mfgpr' for applying Multi-fidelity Gaussian process regression or 'gb_stacked' for applying stacking of fidelities in gradient boosting regressor |

## 5. Output

The output is default reports.csv file from the Vega library.
