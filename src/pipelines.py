from functools import partial

from . import pipeline_blocks as blocks


def lightGBM_v1(config, train_mode, suffix=''):
    if train_mode:
        persist_output = True
        cache_output = True
        load_persisted_output = True
    else:
        persist_output = False
        cache_output = False
        load_persisted_output = False

    data_cleaned = blocks.data_cleaning_v1(config, train_mode, suffix,
                                           persist_output=persist_output,
                                           cache_output=cache_output,
                                           load_persisted_output=load_persisted_output)
    features = blocks.feature_extraction_v1(data_cleaned, config,
                                            train_mode, suffix,
                                            persist_output=persist_output,
                                            cache_output=cache_output,
                                            load_persisted_output=load_persisted_output)
    light_gbm = blocks.classifier_light_gbm(features, config, train_mode, suffix,
                                            persist_output=persist_output,
                                            cache_output=cache_output,
                                            load_persisted_output=load_persisted_output)
    return light_gbm


def lightGBM_v2(config, train_mode, suffix='', use_imputed=True, use_is_missing=True):
    if train_mode:
        persist_output = True
        cache_output = True
        load_persisted_output = True
    else:
        persist_output = False
        cache_output = False
        load_persisted_output = False

    data_cleaned = blocks.data_cleaning_v2(config, train_mode, suffix,
                                           persist_output=persist_output,
                                           cache_output=cache_output,
                                           load_persisted_output=load_persisted_output)
    features = blocks.feature_extraction_v2(data_cleaned, config,
                                            train_mode, suffix,
                                            use_imputed=use_imputed, use_is_missing=use_is_missing,
                                            persist_output=persist_output,
                                            cache_output=cache_output,
                                            load_persisted_output=load_persisted_output)
    light_gbm = blocks.classifier_light_gbm(features, config, train_mode, suffix,
                                            persist_output=persist_output,
                                            cache_output=cache_output,
                                            load_persisted_output=load_persisted_output)
    return light_gbm


PIPELINES = {'lightGBM': lightGBM_v1,
             'lightGBM_is_missing': partial(lightGBM_v2, use_imputed=False, use_is_missing=True),
             'lightGBM_imputed': partial(lightGBM_v2, use_imputed=True, use_is_missing=False),
             'lightGBM_imputed_and_is_missing': partial(lightGBM_v2, use_imputed=True, use_is_missing=True),
             }
