# Human Language Technology Project UniPi 2022/2023 
## Legal Judgement Predictor for ECHR
![echr](images/COE-logo-ECHR.png)

This repository contains the code for the Human Language Technology Project of the University of Pisa, academic year 2022/2023. The project consists in the development of a Legal Judgement Predictor for the European Court of Human Rights (ECHR). The dataset used for the project is the [ECHR dataset][def1] which was made publicily available by the authors of the paper [Neural Legal Judgement Prediction in English][def2].

## Introduction
This project aims to focus on some of the task presented in the paper [Neural Legal Judgement Prediction in English][def2]. In particular, the project focuses on the task of predicting the judgement of a case given its description. Our idea is to use different LLM model such as BERT and compare their performance. Also we want to try to train the LLM in order to give a proper sentence with justification for the judgement, not only the binary output.

## Dataset
The dataset is composed of 5847 cases, each of which is composed of a text and a label. The text is the description of the case, while the label is the judgement of the case. The judgement can be either "violation" or "non-violation". The dataset is divided into 3 subsets: train, dev and test. The train set is composed of 7100 cases, the dev set is composed of 1380 cases and the test set is composed of 2998 cases.

## Task
The task is to predict the judgement of a case given its description. The task can be binary or multi-class. In the binary case the output is either "violation" or "non-violation".

## Authors
- [Giacomo Lagomarsini](https://github.com/g-lago8)
- [Leonardo Stoppani](https://github.com/lilf4p)

## References
- [Neural Legal Judgement Prediction in English][def2]
- [ECHR dataset][def1]

[def1]: https://archive.org/details/ECHR-ACL2019
[def2]: https://aclanthology.org/P19-1424.pdf
