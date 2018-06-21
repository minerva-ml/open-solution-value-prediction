# Santander Value Prediction Challenge: Open Solution

[![Join the chat at https://gitter.im/minerva-ml/open-solution-value-prediction](https://badges.gitter.im/minerva-ml/open-solution-value-prediction.svg)](https://gitter.im/minerva-ml/open-solution-value-prediction?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This is an open solution to the [Santander Value Prediction Challenge](https://www.kaggle.com/c/santander-value-prediction-challenge).

## The purpose of the Open Solution
We are building entirely open solution to this competition. Specifically:
1. Check **live preview of our work** on public projects page: [Santander Value Prediction Challenge](https://app.neptune.ml/neptune-ml/Santander-Value-Prediction-Challenge).
1. Source code and [issues](https://github.com/minerva-ml/open-solution-value-prediction/issues) are publicly available.

Rules are simple:
1. Clean code and extensible solution leads to the reproducible experimentations and better control over the improvements.
1. Open solution should establish solid benchmark and give good base for your custom ideas and experiments.

## Installation
### Fast Track
1. Clone repository and install requirements (check _requirements.txt_)
1. Register to the [neptune.ml](https://neptune.ml) _(if you wish to use it)_
1. Run experiment:
```bash
neptune run --config neptune_random_search.yaml main.py train_evaluate_predict --pipeline_name SOME_NAME
```

### Step by step
1. Clone this repository
```bash
git clone https://github.com/minerva-ml/open-solution-value-prediction.git
```
2. Install requirements in your Python3 environment
```bash
pip3 install requirements.txt
```
3. Register to the [neptune.ml](https://neptune.ml) _(if you wish to use it)_
4. Update data directories in the [neptune.yaml](https://github.com/minerva-ml/open-solution-value-prediction/blob/master/neptune.yaml) configuration file
5. Run experiment:
```bash
neptune login
neptune run --config neptune_random_search.yaml main.py train_evaluate_predict --pipeline_name SOME_NAME
```
6. collect submit from `experiment_directory` specified in the [neptune.yaml](https://github.com/minerva-ml/open-solution-value-prediction/blob/master/neptune.yaml)

## Get involved
You are welcome to contribute your code and ideas to this open solution. To get started:
1. Check [competition project](https://github.com/minerva-ml/open-solution-value-prediction/projects/1) on GitHub to see what we are working on right now.
1. Express your interest in paticular task by writing comment in this task, or by creating new one with your fresh idea.
1. We will get back to you quickly in order to start working together.
1. Check [CONTRIBUTING](CONTRIBUTING.md) for some more information.

## User support
There are several ways to seek help:
1. Kaggle [discussion](https://www.kaggle.com/c/santander-value-prediction-challenge) is our primary way of communication.
1. Read project's [Wiki](https://github.com/minerva-ml/open-solution-value-prediction/wiki), where we publish descriptions about the code, pipelines and supporting tools such as [neptune.ml](https://neptune.ml).
1. Submit an [issue]((https://github.com/minerva-ml/open-solution-value-prediction/issues)) directly in this repo.
