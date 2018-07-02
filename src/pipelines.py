from functools import partial

from . import pipeline_blocks as blocks


def lightGBM(config, train_mode, suffix='',
             use_raw=True, use_is_missing=False, use_projections=False, use_row_aggregations=False):
    if train_mode:
        cache_output = False
        persist_output = False
        load_persisted_output = False
    else:
        cache_output = False
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
    if use_row_aggregations:
        row_aggregation_features = blocks.row_aggregation_features(config, train_mode, suffix,
                                                                   persist_output=persist_output,
                                                                   cache_output=cache_output,
                                                                   load_persisted_output=load_persisted_output)
    else:
        row_aggregation_features = None

    features = blocks.feature_extraction(data_cleaned, row_aggregation_features, config,
                                         train_mode, suffix,
                                         use_raw, use_is_missing, use_projections,
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
                                            use_row_aggregations=False),
             'lightGBM_projections': partial(lightGBM,
                                             use_raw=False,
                                             use_is_missing=False,
                                             use_projections=True,
                                             use_row_aggregations=False),
             'lightGBM_aggregations': partial(lightGBM,
                                              use_raw=False,
                                              use_is_missing=False,
                                              use_projections=False,
                                              use_row_aggregations=True),
             'lightGBM_projections_aggregations': partial(lightGBM,
                                                          use_raw=False,
                                                          use_is_missing=False,
                                                          use_projections=True,
                                                          use_row_aggregations=True),
             'lightGBM_raw_projections': partial(lightGBM,
                                                 use_raw=True,
                                                 use_is_missing=False,
                                                 use_projections=True,
                                                 use_row_aggregations=False),
             'lightGBM_raw_aggregations': partial(lightGBM,
                                                  use_raw=True,
                                                  use_is_missing=False,
                                                  use_projections=False,
                                                  use_row_aggregations=True)

             }
