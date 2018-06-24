from functools import partial

import numpy as np
from steppy.adapter import Adapter, E
from steppy.base import Step, BaseTransformer
from toolkit.preprocessing.misc import TruncatedSVD

from . import data_cleaning as dc
from . import feature_extraction as fe
from .hyperparameter_tuning import RandomSearchOptimizer, NeptuneMonitor, PersistResults
from .utils import make_transformer, root_mean_squared_error, to_pandas
from .models import LightGBM


def classifier_light_gbm(features, config, train_mode, suffix='', **kwargs):
    if train_mode:
        features_train, features_valid = features
        log_target = Step(name='log_target{}'.format(suffix),
                          transformer=make_transformer(lambda x: np.log1p(x), output_name='y'),
                          input_data=['input'],
                          adapter=Adapter({'x': E('input', 'y')}),
                          experiment_directory=config.pipeline.experiment_directory, **kwargs)

        log_target_valid = Step(name='log_target_valid{}'.format(suffix),
                                transformer=log_target,
                                input_data=['input'],
                                adapter=Adapter({'x': E('input', 'y_valid')}),
                                experiment_directory=config.pipeline.experiment_directory, **kwargs)

        if config.random_search.light_gbm.n_runs:
            transformer = RandomSearchOptimizer(TransformerClass=LightGBM,
                                                params=config.light_gbm,
                                                train_input_keys=[],
                                                valid_input_keys=['X_valid', 'y_valid'],
                                                score_func=root_mean_squared_error,
                                                maximize=False,
                                                n_runs=config.random_search.light_gbm.n_runs,
                                                callbacks=[
                                                    NeptuneMonitor(
                                                        **config.random_search.light_gbm.callbacks.neptune_monitor),
                                                    PersistResults(
                                                        **config.random_search.light_gbm.callbacks.persist_results)]
                                                )
        else:
            transformer = LightGBM(**config.light_gbm)

        light_gbm = Step(name='light_gbm{}'.format(suffix),
                         transformer=transformer,
                         input_data=['input'],
                         input_steps=[features_train, features_valid, log_target, log_target_valid],
                         adapter=Adapter({'X': E(features_train.name, 'features'),
                                          'y': E(log_target.name, 'y'),
                                          'feature_names': E(features_train.name, 'feature_names'),
                                          'categorical_features': E(features_train.name, 'categorical_features'),
                                          'X_valid': E(features_valid.name, 'features'),
                                          'y_valid': E(log_target_valid.name, 'y'),
                                          }),
                         experiment_directory=config.pipeline.experiment_directory, **kwargs)
    else:
        light_gbm = Step(name='light_gbm{}'.format(suffix),
                         transformer=LightGBM(**config.light_gbm),
                         input_steps=[features],
                         adapter=Adapter({'X': E(features.name, 'features')}),
                         experiment_directory=config.pipeline.experiment_directory, **kwargs)

    output = exp_target(light_gbm, config, suffix, **kwargs)

    return output


def exp_target(model_output, config, suffix, **kwargs):
    exp_target = Step(name='exp_target{}'.format(suffix),
                      transformer=make_transformer(lambda x: np.exp(x) - 1, output_name='prediction'),
                      input_steps=[model_output],
                      adapter=Adapter({'x': E(model_output.name, 'prediction')}),
                      experiment_directory=config.pipeline.experiment_directory, **kwargs)
    return exp_target


def data_cleaning_v1(config, train_mode, suffix, **kwargs):
    drop_constant = Step(name='drop_constant{}'.format(suffix),
                         transformer=dc.VarianceThreshold(**config.variance_threshold),
                         input_data=['input'],
                         experiment_directory=config.pipeline.experiment_directory, **kwargs)

    drop_duplicate = Step(name='drop_duplicate{}'.format(suffix),
                          transformer=dc.DropDuplicateColumns(),
                          input_steps=[drop_constant],
                          experiment_directory=config.pipeline.experiment_directory, **kwargs)

    drop_zero_fraction = Step(name='drop_zero_fraction{}'.format(suffix),
                              transformer=dc.DropOneValueFrequent(**config.drop_zero_fraction),
                              input_steps=[drop_duplicate],
                              experiment_directory=config.pipeline.experiment_directory, **kwargs)
    to_numerical = Step(name='to_numerical{}'.format(suffix),
                        transformer=make_transformer(lambda X: X, output_name='numerical_features'),
                        input_steps=[drop_zero_fraction],
                        experiment_directory=config.pipeline.experiment_directory, **kwargs)

    if train_mode:
        drop_constant_valid = Step(name='drop_constant_valid{}'.format(suffix),
                                   transformer=drop_constant,
                                   input_data=['input'],
                                   adapter=Adapter({'X': E('input', 'X_valid'),
                                                    }
                                                   ),
                                   experiment_directory=config.pipeline.experiment_directory, **kwargs)

        drop_duplicate_valid = Step(name='drop_duplicate_valid{}'.format(suffix),
                                    transformer=drop_duplicate,
                                    input_steps=[drop_constant_valid],
                                    experiment_directory=config.pipeline.experiment_directory, **kwargs)

        drop_zero_fraction_valid = Step(name='drop_zero_fraction_valid{}'.format(suffix),
                                        transformer=drop_zero_fraction,
                                        input_steps=[drop_duplicate_valid],
                                        experiment_directory=config.pipeline.experiment_directory, **kwargs)

        to_numerical_valid = Step(name='to_numerical_valid{}'.format(suffix),
                                  transformer=to_numerical,
                                  input_steps=[drop_zero_fraction_valid],
                                  experiment_directory=config.pipeline.experiment_directory, **kwargs)

        return to_numerical, to_numerical_valid
    else:
        return to_numerical


def data_cleaning_v2(config, train_mode, suffix, **kwargs):
    cleaned_data = data_cleaning_v1(config, train_mode, suffix, **kwargs)

    if train_mode:
        cleaned_data, cleaned_data_valid = cleaned_data

    impute_missing = Step(name='impute_missing{}'.format(suffix),
                          transformer=dc.ImputeMissing(**config.impute_missing),
                          input_steps=[cleaned_data],
                          adapter=Adapter({'X': E(cleaned_data.name, 'numerical_features'),
                                           }
                                          ),
                          experiment_directory=config.pipeline.experiment_directory, **kwargs)

    if train_mode:
        impute_missing_valid = Step(name='impute_missing_valid{}'.format(suffix),
                                    transformer=impute_missing,
                                    input_steps=[cleaned_data_valid],
                                    adapter=Adapter({'X': E(cleaned_data_valid.name, 'numerical_features'),
                                                     }
                                                    ),
                                    experiment_directory=config.pipeline.experiment_directory, **kwargs)
        return impute_missing, impute_missing_valid
    else:
        return impute_missing


def feature_extraction_v1(data_cleaned, config, train_mode, suffix, **kwargs):
    if train_mode:
        data_cleaned_train, data_cleaned_valid = data_cleaned
        feature_combiner, feature_combiner_valid = _join_features(numerical_features=[data_cleaned_train],
                                                                  numerical_features_valid=[data_cleaned_valid],
                                                                  categorical_features=[],
                                                                  categorical_features_valid=[],
                                                                  config=config,
                                                                  train_mode=train_mode,
                                                                  suffix=suffix, **kwargs)
        return feature_combiner, feature_combiner_valid
    else:
        feature_combiner = _join_features(numerical_features=[data_cleaned],
                                          numerical_features_valid=[],
                                          categorical_features=[],
                                          categorical_features_valid=[],
                                          config=config,
                                          train_mode=train_mode,
                                          suffix=suffix, **kwargs)

        return feature_combiner


def feature_extraction_v2(data_cleaned, config, train_mode, suffix, use_imputed, use_is_missing, **kwargs):
    if train_mode:
        data_cleaned_train, data_cleaned_valid = data_cleaned

        numerical_features, categorical_features = [], []
        numerical_features_valid, categorical_features_valid = [], []
        if use_imputed:
            numerical_features.append(data_cleaned_train)
            numerical_features_valid.append(data_cleaned_valid)
        elif use_is_missing:
            categorical_features.append(data_cleaned_train)
            categorical_features_valid.append(data_cleaned_valid)

        print(numerical_features, categorical_features)
        feature_combiner, feature_combiner_valid = _join_features(numerical_features=numerical_features,
                                                                  numerical_features_valid=numerical_features_valid,
                                                                  categorical_features=categorical_features,
                                                                  categorical_features_valid=categorical_features_valid,
                                                                  config=config,
                                                                  train_mode=train_mode,
                                                                  suffix=suffix, **kwargs)
        return feature_combiner, feature_combiner_valid
    else:
        numerical_features, categorical_features = [], []
        if use_imputed:
            numerical_features.append(data_cleaned)
        elif use_is_missing:
            categorical_features.append(data_cleaned)

        feature_combiner = _join_features(numerical_features=numerical_features,
                                          numerical_features_valid=[],
                                          categorical_features=categorical_features,
                                          categorical_features_valid=[],
                                          config=config,
                                          train_mode=train_mode,
                                          suffix=suffix, **kwargs)

        return feature_combiner


def _join_features(numerical_features,
                   numerical_features_valid,
                   categorical_features,
                   categorical_features_valid,
                   config, train_mode, suffix, **kwargs):
    feature_joiner = Step(name='feature_joiner{}'.format(suffix),
                          transformer=fe.FeatureJoiner(),
                          input_steps=numerical_features + categorical_features,
                          adapter=Adapter({
                              'numerical_feature_list': [
                                  E(feature.name, 'numerical_features') for feature in numerical_features],
                              'categorical_feature_list': [
                                  E(feature.name, 'categorical_features') for feature in categorical_features],
                          }),
                          experiment_directory=config.pipeline.experiment_directory,
                          **kwargs)

    if train_mode:
        feature_joiner_valid = Step(name='feature_joiner_valid{}'.format(suffix),
                                    transformer=feature_joiner,
                                    input_steps=numerical_features_valid + categorical_features_valid,
                                    adapter=Adapter({
                                        'numerical_feature_list': [
                                            E(feature.name,
                                              'numerical_features') for feature in numerical_features_valid],
                                        'categorical_feature_list': [
                                            E(feature.name,
                                              'categorical_features') for feature in categorical_features_valid],
                                    }),
                                    experiment_directory=config.pipeline.experiment_directory,
                                    **kwargs)

        return feature_joiner, feature_joiner_valid

    else:
        return feature_joiner
