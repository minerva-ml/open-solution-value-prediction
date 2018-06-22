from functools import partial

import numpy as np
from steppy.adapter import Adapter, E
from steppy.base import Step, BaseTransformer
from toolkit.preprocessing.misc import TruncatedSVD

from . import feature_extraction as fe
from .feature_selection import DropDuplicateColumns, VarianceThreshold, LassoFeatureSelection
from .hyperparameter_tuning import RandomSearchOptimizer, NeptuneMonitor, PersistResults
from .utils import make_transformer, root_mean_squared_error, to_pandas
from .models import LightGBM


def classifier_light_gbm(features, config, train_mode, suffix='', **kwargs):
    if train_mode:
        features_train, features_valid = features
        log_target = Step(name='log_target{}'.format(suffix),
                          transformer=make_transformer(lambda x: np.log(x + 1), output_name='y'),
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


def data_cleaning(config, train_mode, suffix, **kwargs):
    drop_constant = Step(name='drop_constant{}'.format(suffix),
                         transformer=VarianceThreshold(**config.variance_threshold),
                         input_data=['input'],
                         adapter=Adapter({'X': E('input', 'X')
                                          }
                                         ),
                         experiment_directory=config.pipeline.experiment_directory, **kwargs)

    drop_duplicate = Step(name='drop_duplicate{}'.format(suffix),
                          transformer=DropDuplicateColumns(),
                          input_steps=[drop_constant],
                          adapter=Adapter({'X': E(drop_constant.name, 'X'),
                                           }
                                          ),
                          experiment_directory=config.pipeline.experiment_directory, **kwargs)

    log_num = Step(name='log_num{}'.format(suffix),
                   transformer=make_transformer(lambda x: np.log(x + 1), output_name='numerical_features'),
                   input_steps=[drop_duplicate],
                   adapter=Adapter({'x': E(drop_duplicate.name, 'X')}
                                   ),
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
                                    adapter=Adapter({'X': E(drop_constant_valid.name, 'X'),
                                                     }
                                                    ),
                                    experiment_directory=config.pipeline.experiment_directory, **kwargs)

        log_num_valid = Step(name='log_num_valid{}'.format(suffix),
                             transformer=log_num,
                             input_steps=[drop_duplicate],
                             adapter=Adapter({'x': E(drop_duplicate.name, 'X')}
                                             ),
                             experiment_directory=config.pipeline.experiment_directory, **kwargs)

        return drop_duplicate, drop_duplicate_valid
    else:
        return drop_duplicate


def feature_extraction(data_cleaned, config, train_mode, suffix, **kwargs):
    feature_decomposers = [
        (TruncatedSVD, config.truncated_svd, 'trunc_svd'),
        (fe.PCA, config.pca, 'pca'),
        (fe.FastICA, config.fast_ica, 'fast_ica'),
        (fe.FactorAnalysis, config.factor_analysis, 'factor_analysis'),
        (fe.GaussianRandomProjection, config.gaussian_random_projection, 'grp'),
        (fe.SparseRandomProjection, config.sparse_random_projection, 'srp'),
    ]

    if train_mode:
        decomposed_features, decomposed_features_valid = [], []
        for decomposer in feature_decomposers:
            decomposed_feature, decomposed_feature_valid = _decomposition(decomposer, data_cleaned, config, train_mode,
                                                                          suffix)
            decomposed_features.append(decomposed_feature)
            decomposed_features_valid.append(decomposed_feature_valid)

        numerical_features = decomposed_features
        numerical_features_valid = decomposed_features_valid
        feature_combiner, feature_combiner_valid = _join_features(numerical_features=numerical_features,
                                                                  numerical_features_valid=numerical_features_valid,
                                                                  categorical_features=[],
                                                                  categorical_features_valid=[
                                                                  ],
                                                                  config=config,
                                                                  train_mode=train_mode,
                                                                  suffix=suffix, **kwargs)
        return feature_combiner, feature_combiner_valid
    else:
        decomposed_features = []
        for decomposer in feature_decomposers:
            decomposed_feature = _decomposition(decomposer, data_cleaned, config, train_mode,
                                                suffix)
            decomposed_features.append(decomposed_feature)
        numerical_features = decomposed_features

        feature_combiner = _join_features(numerical_features=numerical_features,
                                          numerical_features_valid=[],
                                          categorical_features=[],
                                          categorical_features_valid=[],
                                          config=config,
                                          train_mode=train_mode,
                                          suffix=suffix, **kwargs)

        return feature_combiner


def feature_selection(features, config, train_mode, suffix, **kwargs):
    """
    Todo:
        Right now no lasso-based feature selection is performed
    """
    if train_mode:
        features, features_valid = features

    if train_mode:
        lasso_selector = Step(name='lasso_selector{}'.format(suffix),
                              transformer=LassoFeatureSelection(**config.lasso_feature_selector),
                              input_data=['input'],
                              input_steps=[features],
                              adapter=Adapter({'target': E('input', 'y'),
                                               'features': E(features.name, 'features'),
                                               'feature_names': E(features.name, 'feature_names'),
                                               'categorical_features': E(features.name, 'categorical_features')
                                               }
                                              ),
                              experiment_directory=config.pipeline.experiment_directory, **kwargs)

        lasso_selector_valid = Step(name='lasso_selector_valid{}'.format(suffix),
                                    transformer=lasso_selector,
                                    input_steps=[features_valid],
                                    adapter=Adapter({'features': E(features_valid.name, 'features')}
                                                    ),
                                    experiment_directory=config.pipeline.experiment_directory, **kwargs)

        return features, features_valid
    else:
        lasso_selector = Step(name='lasso_selector{}'.format(suffix),
                              transformer=LassoFeatureSelection(**config.lasso_feature_selector),
                              input_steps=[features],
                              adapter=Adapter({'features': E(features.name, 'features')}
                                              ),
                              experiment_directory=config.pipeline.experiment_directory, **kwargs)

        return features


def _feature_by_type_splits(features_cleaned, config, train_mode, suffix):
    if train_mode:
        features_cleaned, features_cleaned_valid = features_cleaned
    feature_by_type_split = Step(name='inferred_type_splitter{}'.format(suffix),
                                 transformer=fe.InferredTypeSplitter(),
                                 input_steps=[features_cleaned],
                                 adapter=Adapter({'X': E(features_cleaned.name, 'X')}),
                                 experiment_directory=config.pipeline.experiment_directory)

    if train_mode:
        feature_by_type_split_valid = Step(name='inferred_type_splitter_valid{}'.format(suffix),
                                           transformer=feature_by_type_split,
                                           input_steps=[features_cleaned_valid],
                                           adapter=Adapter({'X': E(features_cleaned_valid.name, 'X')}),
                                           experiment_directory=config.pipeline.experiment_directory)
        return feature_by_type_split, feature_by_type_split_valid
    else:
        return feature_by_type_split


def _decomposition(decomposer_config, data_cleaned, config, train_mode, suffix, **kwargs):
    (DecompositionTransformer, transformer_config, transformer_name) = decomposer_config

    if train_mode:
        data_cleaned, data_cleaned_valid = data_cleaned

    decomposer = Step(name='{}{}'.format(transformer_name, suffix),
                      transformer=DecompositionTransformer(**transformer_config),
                      input_steps=[data_cleaned],
                      adapter=Adapter({'features': E(data_cleaned.name, 'X')}),
                      experiment_directory=config.pipeline.experiment_directory, **kwargs)

    decomposer_pandas = Step(name='{}_pandas{}'.format(transformer_name, suffix),
                             transformer=make_transformer(partial(to_pandas, column_prefix=transformer_name)
                                                          , output_name='numerical_features'),
                             input_steps=[decomposer],
                             adapter=Adapter({'x': E(decomposer.name, 'features')}),
                             experiment_directory=config.pipeline.experiment_directory, **kwargs)

    if train_mode:
        decomposer_valid = Step(name='{}_valid{}'.format(transformer_name, suffix),
                                transformer=decomposer,
                                input_steps=[data_cleaned_valid],
                                adapter=Adapter({'features': E(data_cleaned_valid.name, 'X')}
                                                ),
                                experiment_directory=config.pipeline.experiment_directory, **kwargs)
        decomposer_pandas_valid = Step(name='{}_pandas_valid{}'.format(transformer_name, suffix),
                                       transformer=decomposer_pandas,
                                       input_steps=[decomposer_valid],
                                       adapter=Adapter({'x': E(decomposer_valid.name, 'features')}),
                                       experiment_directory=config.pipeline.experiment_directory, **kwargs)
        return decomposer_pandas, decomposer_pandas_valid
    else:
        return decomposer_pandas


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
