from functools import partial

from . import pipeline_blocks as blocks


def lightGBM(config, train_mode, suffix='',
             use_raw=True, use_is_missing=False, use_projections=False, use_aggregations=False):
    """
    Cleaning:
        - drop all constant columns
        - drop duplicate columns
        - drop almost all zero set drop_zero_fraction__threshold parameter
        - 0s treated as missing values (questionable)

    Feature Extraction:
        - is missing dummy table

    Model:
        - lighbgbm
        - all params in neptune.yaml
        - lightgbm trained on just dummy is_missing table gets 1.51 CV 1.77 LB, interesting
    """
    if train_mode:
        cache_output = True
        persist_output = True
        load_persisted_output = True
    else:
        cache_output = True
        persist_output = False
        load_persisted_output = False

    if use_is_missing:
        cleaning_pipeline = blocks.data_cleaning_v2
    else:
        cleaning_pipeline = blocks.data_cleaning_v1

    data_cleaned = cleaning_pipeline(config, train_mode, suffix,
                                     persist_output=persist_output,
                                     cache_output=cache_output,
                                     load_persisted_output=load_persisted_output)
    features = blocks.feature_extraction(data_cleaned, config,
                                         train_mode, suffix,
                                         use_raw, use_is_missing, use_projections, use_aggregations,
                                         persist_output=persist_output,
                                         cache_output=cache_output,
                                         load_persisted_output=load_persisted_output)
    light_gbm = blocks.classifier_light_gbm(features, config, train_mode, suffix,
                                            persist_output=persist_output,
                                            cache_output=cache_output,
                                            load_persisted_output=load_persisted_output)
    return light_gbm


PIPELINES = {'lightGBM_raw': lightGBM,
             'lightGBM_is_missing': partial(lightGBM,
                                            use_raw=False,
                                            use_is_missing=True,
                                            use_projections=False,
                                            use_aggregations=False),
             'lightGBM_projections': partial(lightGBM,
                                             use_raw=False,
                                             use_is_missing=False,
                                             use_projections=True,
                                             use_aggregations=False),
             'lightGBM_aggregations': partial(lightGBM,
                                              use_raw=False,
                                              use_is_missing=False,
                                              use_projections=False,
                                              use_aggregations=True),
             'lightGBM_raw_projections': partial(lightGBM,
                                                 use_raw=True,
                                                 use_is_missing=False,
                                                 use_projections=True,
                                                 use_aggregations=False),
             'lightGBM_raw_aggregations': partial(lightGBM,
                                                  use_raw=True,
                                                  use_is_missing=False,
                                                  use_projections=False,
                                                  use_aggregations=True),
             'lightGBM_raw_projections_aggregations': partial(lightGBM,
                                                              use_raw=True,
                                                              use_is_missing=False,
                                                              use_projections=True,
                                                              use_aggregations=True),

             }
