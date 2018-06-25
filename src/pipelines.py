from functools import partial

from . import pipeline_blocks as blocks


def lightGBM_v1(config, train_mode, suffix=''):
    """
    Cleaning:
        - drop all constant columns
        - drop duplicate columns
        - drop almost all zero set drop_zero_fraction__threshold parameter

    Feature Extraction:
        - just treat all values as numerical

    Model:
        - lighbgbm
        - all params in neptune.yaml
        - 0s treated as missing value 1.39 CV 1.43 LB
    """
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


def lightGBM_v2(config, train_mode, suffix=''):
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

    data_cleaned = blocks.data_cleaning_v2(config, train_mode, suffix,
                                           persist_output=persist_output,
                                           cache_output=cache_output,
                                           load_persisted_output=load_persisted_output)
    features = blocks.feature_extraction_v2(data_cleaned, config,
                                            train_mode, suffix,
                                            persist_output=persist_output,
                                            cache_output=cache_output,
                                            load_persisted_output=load_persisted_output)
    light_gbm = blocks.classifier_light_gbm(features, config, train_mode, suffix,
                                            persist_output=persist_output,
                                            cache_output=cache_output,
                                            load_persisted_output=load_persisted_output)
    return light_gbm


def lightGBM_v3(config, train_mode, suffix=''):
    """
    Cleaning:
        - drop all constant columns
        - drop duplicate columns
        - drop almost all zero set drop_zero_fraction__threshold parameter

    Feature Extraction:
        - All features are projections to lower dimensional space
        - Truncated svd: set truncated_svd__n_components
        - PCA: set pca__n_components
        - Fast ICA set fast_ica__n_components
        - Factor Analysis set factor_analysis__n_components
        - Gaussian random projections set gaussian_random_projection__n_components and gaussian_projection__eps
        - Sparse random projection set sparse_random_projection__n_components

    Model:
        - lighbgbm
        - all params in neptune.yaml
        - Truncated svd: alone gives 1.56 CV
        - PCA: alone gives 1.55 CV
        - Fast ICA: alone gives XXX CV
        - Factor Analysis: alone gives XXX CV
        - Gaussian random projections: alone gives XXX CV
        - Sparse random projection: alone gives XXX CV
        """
    if train_mode:
        cache_output = True
        persist_output = True
        load_persisted_output = True
    else:
        cache_output = True
        persist_output = False
        load_persisted_output = False

    data_cleaned = blocks.data_cleaning_v1(config, train_mode, suffix,
                                           persist_output=persist_output,
                                           cache_output=cache_output,
                                           load_persisted_output=load_persisted_output)
    features = blocks.feature_extraction_v3(data_cleaned, config,
                                            train_mode, suffix,
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
             'lightGBM_projections': lightGBM_v3,

             }
