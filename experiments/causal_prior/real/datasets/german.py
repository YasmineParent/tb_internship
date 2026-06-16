"""German Credit (Statlog) loader, via openml ('credit-g').

n=1000, the canonical interpretable credit-scorecard benchmark and a strong
mixed-data stress test: 7 numeric/ordinal features and 13 categorical ones
(one-hot). target = credit risk; positive class is 'bad' (the risk event), matching
the FICO convention.
"""
import numpy as np

NUM = ['duration', 'credit_amount', 'installment_commitment', 'residence_since',
       'age', 'existing_credits', 'num_dependents']


def load(args=None):
    from sklearn.datasets import fetch_openml
    d = fetch_openml('credit-g', version=1, as_frame=True)
    df, target = d.data, d.target
    cat = [c for c in df.columns if c not in NUM]
    out, names = [], []
    for c in NUM:
        out.append(df[c].to_numpy(float)); names.append(c)
    for c in cat:
        for lv in sorted(df[c].astype(str).unique()):
            out.append((df[c].astype(str) == lv).to_numpy(float)); names.append(f'{c}={lv}')
    y = np.where(target.to_numpy() == 'bad', 1, -1).astype(int)
    return np.column_stack(out), np.asarray(names), y
