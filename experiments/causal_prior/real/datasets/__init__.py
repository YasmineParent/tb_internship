"""real-data loaders, one common interface so any runner takes --dataset.

each module exposes `load(args) -> (X_orig, names, y)`: X_orig is the original
(pre-binarization) feature matrix, names a 1d array of feature names, y in {-1,1}.
categoricals are one-hot to {0,1} indicators so the conditional-gaussian discovery
treats them as factors and the binarizer passes them through. dataset-specific
options are read off the args namespace (e.g. fico's --sentinel-nan).
"""
from . import adult, fico, german, heart, hepatitis, ilpd, mammographic, tb

DATASETS = {'adult': adult, 'fico': fico, 'german': german, 'heart': heart,
            'hepatitis': hepatitis, 'ilpd': ilpd, 'mammographic': mammographic, 'tb': tb}


def load_dataset(name, args):
    if name not in DATASETS:
        raise ValueError(f'unknown dataset {name!r}; have {sorted(DATASETS)}')
    return DATASETS[name].load(args)
