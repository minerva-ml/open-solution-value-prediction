from functools import partial

from .pipeline_blocks import data_cleaning, feature_extraction, feature_selection, classifier_light_gbm


def lightGBM(config, train_mode, suffix=''):
    if train_mode:
        persist_output = True
        cache_output = True
        load_persisted_output = True
    else:
        persist_output = False
        cache_output = False
        load_persisted_output = False

    data_cleaned = data_cleaning(config, train_mode, suffix,
                                 persist_output=persist_output,
                                 cache_output=cache_output,
                                 load_persisted_output=load_persisted_output)
    features = feature_extraction(data_cleaned, config,
                                  train_mode, suffix,
                                  persist_output=persist_output,
                                  cache_output=cache_output,
                                  load_persisted_output=load_persisted_output)
    selected_features = feature_selection(features, config, train_mode, suffix,
                                          persist_output=persist_output,
                                          cache_output=cache_output,
                                          load_persisted_output=load_persisted_output)
    light_gbm = classifier_light_gbm(selected_features, config, train_mode, suffix,
                                     persist_output=persist_output,
                                     cache_output=cache_output,
                                     load_persisted_output=load_persisted_output)
    return light_gbm


PIPELINES = {'lightGBM': lightGBM
             }
