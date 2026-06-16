"""shared openml loader for mixed tabular datasets.

numeric columns are kept (median-imputed); object/category columns are one-hot to
{0,1}, with missing kept as its own '=nan' indicator (informative missingness).
rows with a missing target are dropped. y = +1 where target == pos_label.
"""
import numpy as np
import pandas as pd


def load_mixed(name, version, pos_label):
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name, version=version, as_frame=True)
    df, target = d.data, d.target.astype(str)
    keep = target != 'nan'
    df, target = df[keep.to_numpy()].reset_index(drop=True), target[keep.to_numpy()].reset_index(drop=True)
    out, names = [], []
    for c in df.columns:
        s = df[c]
        if str(s.dtype) in ('object', 'category'):
            # to object first so category NaN becomes the 'nan' indicator, not a float
            s2 = s.astype('object').fillna('nan').astype(str)
            for lv in sorted(s2.unique()):
                out.append((s2 == lv).to_numpy(float)); names.append(f'{c}={lv}')
        else:
            col = pd.to_numeric(s, errors='coerce')
            out.append(col.fillna(col.median()).to_numpy(float)); names.append(str(c))
    y = np.where(target.to_numpy() == pos_label, 1, -1).astype(int)
    return np.column_stack(out), np.asarray(names), y
