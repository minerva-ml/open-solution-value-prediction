import numpy as np
from steppy.adapter import Adapter, E
from steppy.base import Step, BaseTransformer

from . import feature_extraction as fe
from .feature_selection import LassoFeatureSelection
from .hyperparameter_tuning import RandomSearchOptimizer, NeptuneMonitor, PersistResults
from .utils import make_transformer, root_mean_squared_error
from .models import LightGBM


def classifier_light_gbm(features, config, train_mode, suffix='', **kwargs):
    if train_mode:

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

        features_train, features_valid = features

        selected_features, selected_features_valid = _select_features((features_train, features_valid),
                                                                      config, train_mode, suffix, **kwargs)

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
                         input_steps=[selected_features, selected_features_valid, log_target, log_target_valid],
                         adapter=Adapter({'X': E(selected_features.name, 'features'),
                                          'y': E(log_target.name, 'y'),
                                          'feature_names': E(selected_features.name, 'feature_names'),
                                          'categorical_features': E(selected_features.name, 'categorical_features'),
                                          'X_valid': E(selected_features_valid.name, 'features'),
                                          'y_valid': E(log_target_valid.name, 'y'),
                                          }),
                         experiment_directory=config.pipeline.experiment_directory,
                         **kwargs)
    else:
        selected_features = _select_features(features, config, train_mode, suffix, **kwargs)

        light_gbm = Step(name='light_gbm{}'.format(suffix),
                         transformer=LightGBM(**config.light_gbm),
                         input_steps=[selected_features],
                         adapter=Adapter({'X': E(selected_features.name, 'features')}),
                         experiment_directory=config.pipeline.experiment_directory,
                         **kwargs)

    output = exp_target(light_gbm, config, suffix, **kwargs)

    return output


def exp_target(model_output, config, suffix, **kwargs):
    exp_target = Step(name='exp_target{}'.format(suffix),
                      transformer=make_transformer(lambda x: np.exp(x) - 1, output_name='prediction'),
                      input_steps=[model_output],
                      adapter=Adapter({'x': E(model_output.name, 'prediction')}),
                      experiment_directory=config.pipeline.experiment_directory, **kwargs)
    return exp_target


def feature_extraction(config, train_mode, suffix, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = _feature_by_type_splits(config, train_mode, suffix)

        log_num, log_num_valid = _numerical_transforms((feature_by_type_split, feature_by_type_split_valid),
                                                       config, train_mode, suffix)

        feature_combiner, feature_combiner_valid = _join_features(numerical_features=[log_num],
                                                                  numerical_features_valid=[log_num_valid],
                                                                  categorical_features=[feature_by_type_split],
                                                                  categorical_features_valid=[
                                                                      feature_by_type_split_valid],
                                                                  config=config,
                                                                  train_mode=train_mode,
                                                                  suffix=suffix, **kwargs)
        return feature_combiner, feature_combiner_valid
    else:
        feature_by_type_split = _feature_by_type_splits(config, train_mode, suffix)

        log_num = _numerical_transforms(feature_by_type_split, config, train_mode, suffix)

        feature_combiner = _join_features(numerical_features=[log_num],
                                          numerical_features_valid=[],
                                          categorical_features=[feature_by_type_split],
                                          categorical_features_valid=[],
                                          config=config,
                                          train_mode=train_mode,
                                          suffix=suffix, **kwargs)

        return feature_combiner


def _feature_by_type_splits(config, train_mode, suffix):
    if train_mode:
        feature_by_type_split = Step(name='inferred_type_splitter{}'.format(suffix),
                                     transformer=fe.InferredTypeSplitter(),
                                     input_data=['input'],
                                     adapter=Adapter({'X': E('input', 'X')}),
                                     experiment_directory=config.pipeline.experiment_directory)

        feature_by_type_split_valid = Step(name='inferred_type_splitter_valid{}'.format(suffix),
                                           transformer=feature_by_type_split,
                                           input_data=['input'],
                                           adapter=Adapter({'X': E('input', 'X_valid')}),
                                           experiment_directory=config.pipeline.experiment_directory)

        return feature_by_type_split, feature_by_type_split_valid

    else:
        feature_by_type_split = Step(name='inferred_type_splitter{}'.format(suffix),
                                     transformer=fe.InferredTypeSplitter(),
                                     input_data=['input'],
                                     adapter=Adapter({'X': E('input', 'X')}),
                                     experiment_directory=config.pipeline.experiment_directory)

    return feature_by_type_split


def _numerical_transforms(dispatchers, config, train_mode, suffix, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
    else:
        feature_by_type_split = dispatchers

    log_num = Step(name='log_num{}'.format(suffix),
                   transformer=make_transformer(lambda x: np.log(x + 1), output_name='numerical_features'),
                   input_steps=[feature_by_type_split],
                   adapter=Adapter({'x': E(feature_by_type_split.name, 'numerical_features')}
                                   ),
                   experiment_directory=config.pipeline.experiment_directory, **kwargs)

    if train_mode:
        log_num_valid = Step(name='log_num_valid{}'.format(suffix),
                             transformer=log_num,
                             input_steps=[feature_by_type_split_valid],
                             adapter=Adapter({'x': E(feature_by_type_split_valid.name, 'numerical_features')}
                                             ),
                             experiment_directory=config.pipeline.experiment_directory, **kwargs)
        return log_num, log_num_valid
    else:
        return log_num


def _select_features(features, config, train_mode, suffix, **kwargs):
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

        return lasso_selector, lasso_selector_valid
    else:
        lasso_selector = Step(name='lasso_selector{}'.format(suffix),
                              transformer=LassoFeatureSelection(**config.lasso_feature_selector),
                              input_steps=[features],
                              adapter=Adapter({'features': E(features.name, 'features')}
                                              ),
                              experiment_directory=config.pipeline.experiment_directory, **kwargs)

        return lasso_selector


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
