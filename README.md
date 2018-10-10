# Santander Value Prediction Challenge: Open Solution

[![Join the chat at https://gitter.im/minerva-ml/open-solution-value-prediction](https://badges.gitter.im/minerva-ml/open-solution-value-prediction.svg)](https://gitter.im/minerva-ml/open-solution-value-prediction?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/minerva-ml/open-solution-value-prediction/blob/master/LICENSE)

This is an open solution to the [Santander Value Prediction Challenge](https://www.kaggle.com/c/santander-value-prediction-challenge) :smiley:

## More competitions :sparkler:
Check collection of [public projects :gift:](https://app.neptune.ml/-/explore), where you can find multiple Kaggle competitions with code, experiments and outputs.

## Our goals
We are building entirely open solution to this competition. Specifically:
1. **Learning from the process** - updates about new ideas, code and experiments is the best way to learn data science. Our activity is especially useful for people who wants to enter the competition, but lack appropriate experience.
1. Encourage more Kagglers to start working on this competition.
1. Deliver open source solution with no strings attached. Code is available on our [GitHub repository](https://github.com/neptune-ml/open-solution-value-prediction) :computer:. This solution should establish solid benchmark, as well as provide good base for your custom ideas and experiments. We care about clean code :smiley:
1. We are opening our experiments as well: everybody can have **live preview** on our experiments, parameters, code, etc. Check: [Santander-Value-Prediction-Challenge :chart_with_upwards_trend:](https://app.neptune.ml/neptune-ml/Santander-Value-Prediction-Challenge) and screens below.

| LightGBM train and validation performance on folds :bar_chart: | LightGBM experiment logged values :bar_chart: |
|:---|:---|
|[![train-validation-results-on-folds](https://gist.githubusercontent.com/kamil-kaczmarek/b3b939797fb39752c45fdadfedba3ed9/raw/159817acd944dbd3b5f9f55bdcc0749c18ce1b72/vp-1.png)](https://app.neptune.ml/-/dashboard/experiment/b540e70a-d7e5-444e-bd6f-a56d151ed61f) | [![LightGBM-learning-curves](https://gist.githubusercontent.com/kamil-kaczmarek/b3b939797fb39752c45fdadfedba3ed9/raw/159817acd944dbd3b5f9f55bdcc0749c18ce1b72/vp-2.png)](https://app.neptune.ml/-/dashboard/experiment/b540e70a-d7e5-444e-bd6f-a56d151ed61f) |

## Disclaimer
In this open source solution you will find references to the [neptune.ml](https://neptune.ml). It is free platform for community Users, which we use daily to keep track of our experiments. Please note that using neptune.ml is not necessary to proceed with this solution. You may run it as plain Python script :wink:.

# How to start?
## Learn more about our solutions
1. Check [Kaggle discussion](https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/59299) for most recent updates and comments.
1. Read [Wiki pages](https://github.com/neptune-ml/open-solution-value-prediction/wiki), where we describe solutions in more detail. Click on the tropical fish to get started [:tropical_fish:](https://github.com/neptune-ml/open-solution-value-prediction/wiki) or pick solution from the table below.

|link to code|name|CV|LB|link to the description|
|:---|:---|:---|:---|:---|
|[solution 1](https://github.com/neptune-ml/open-solution-value-prediction/tree/solution-1)|*honey bee* :honeybee:|1.39|1.43|[LightGBM and 5fold CV](LightGBM-and-5fold-CV)|
|[solution 2](https://github.com/neptune-ml/open-solution-value-prediction/tree/solution-2)|*beetle* :beetle:|1.60|1.77|[LightGBM on binarized dataset](LightGBM-on-binarized-dataset)|
|[solution 3](https://github.com/neptune-ml/open-solution-value-prediction/tree/solution-3)|*dromedary camel* :dromedary_camel:|1.35|1.41|[LightGBM with row aggregations](LightGBM-with-row-aggregations)|
|[solution 4](https://github.com/neptune-ml/open-solution-value-prediction/tree/solution-6)|*whale* :whale:|1.3416|1.41|[LightGBM on dimension reduced dataset](LightGBM-on-dimension-reduced-dataset)|
|[solution 5](https://github.com/neptune-ml/open-solution-value-prediction/tree/solution-6)|*water buffalo* :water_buffalo:|1.336|1.39|[Exploring various dimension reduction techniques](Exploring-various-dimension-reduction-techniques)|
|[solution 6](https://github.com/neptune-ml/open-solution-value-prediction/tree/solution-6)|*blowfish* :blowfish:|1.333|1.38|[bucketing row aggregations](bucketing-row-aggregations)|

## Start experimenting with ready-to-use code
You can jump start your participation in the competition by using our starter pack. Installation instruction below will guide you through the setup.

### Installation _(fast track)_
1. Clone repository and install requirements (check _requirements.txt_)
1. Register to the [neptune.ml](https://neptune.ml/login) _(if you wish to use it)_
1. Run experiment:

:trident:
```bash
neptune run --config neptune_random_search.yaml main.py train_evaluate_predict --pipeline_name SOME_NAME
```

:snake:
```bash
python main.py -- train_evaluate_predict --pipeline_name SOME_NAME
```

### Installation _(step by step)_
1. Clone this repository
```bash
git clone https://github.com/minerva-ml/open-solution-value-prediction.git
```
2. Install requirements in your Python3 environment
```bash
pip3 install -r requirements.txt
```
3. Register to the [neptune.ml](https://neptune.ml/login) _(if you wish to use it)_
4. Update data directories in the [neptune.yaml](https://github.com/minerva-ml/open-solution-value-prediction/blob/master/neptune.yaml) configuration file
5. Run experiment:

:trident:
```bash
neptune login
neptune run --config neptune_random_search.yaml main.py train_evaluate_predict --pipeline_name SOME_NAME
```

:snake:
```bash
python main.py -- train_evaluate_predict --pipeline_name SOME_NAME
```

6. collect submit from `experiment_directory` specified in the [neptune.yaml](https://github.com/minerva-ml/open-solution-value-prediction/blob/master/neptune.yaml)

## Get involved
You are welcome to contribute your code and ideas to this open solution. To get started:
1. Check [competition project](https://github.com/minerva-ml/open-solution-value-prediction/projects/1) on GitHub to see what we are working on right now.
1. Express your interest in particular task by writing comment in this task, or by creating new one with your fresh idea.
1. We will get back to you quickly in order to start working together.
1. Check [CONTRIBUTING](CONTRIBUTING.md) for some more information.

## User support
There are several ways to seek help:
1. Kaggle [discussion](https://www.kaggle.com/c/santander-value-prediction-challenge) is our primary way of communication.
1. Read project's [Wiki](https://github.com/minerva-ml/open-solution-value-prediction/wiki), where we publish descriptions about the code, pipelines and supporting tools such as [neptune.ml](https://neptune.ml).
1. Submit an [issue]((https://github.com/minerva-ml/open-solution-value-prediction/issues)) directly in this repo.
