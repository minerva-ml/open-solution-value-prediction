from functools import partial

from .pipeline_blocks import feature_extraction, classifier_light_gbm


def lightGBM(config, train_mode):
    if train_mode:
        features, features_valid = feature_extraction(config, train_mode,
                                                      persist_output=True,
                                                      cache_output=True,
                                                      load_persisted_output=True)
        light_gbm = classifier_light_gbm((features, features_valid), config, train_mode)
    else:
        features = feature_extraction(config, train_mode, cache_output=True)
        light_gbm = classifier_light_gbm(features, config, train_mode)

    return light_gbm


PIPELINES = {'lightGBM': {'train': partial(lightGBM, train_mode=True),
                          'inference': partial(lightGBM, train_mode=False)
                          },
             }
