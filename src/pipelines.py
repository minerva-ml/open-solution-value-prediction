from functools import partial

from .pipeline_blocks import feature_extraction, classifier_light_gbm


def lightGBM(config, train_mode, suffix=''):
    if train_mode:
        features, features_valid = feature_extraction(config, train_mode, suffix,
                                                      persist_output=True,
                                                      cache_output=True,
                                                      load_persisted_output=True)
        light_gbm = classifier_light_gbm((features, features_valid), config, train_mode, suffix)
    else:
        features = feature_extraction(config, train_mode, suffix, cache_output=True)
        light_gbm = classifier_light_gbm(features, config, train_mode, suffix)

    return light_gbm


PIPELINES = {'lightGBM': lightGBM
             }
