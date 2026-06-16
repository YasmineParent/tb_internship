"""Heart Disease (Statlog) loader, via openml.

n=270, mixed: continuous vitals + binary + categorical (chest pain, resting ecg,
slope, thal). categoricals are one-hot to {0,1} indicators. binary outcome
(heart disease present). small, clinical, the natural home for an interpretable
risk score.
"""
import numpy as np

CONT = ['age', 'resting_blood_pressure', 'serum_cholestoral',
        'maximum_heart_rate_achieved', 'oldpeak', 'number_of_major_vessels']
BIN = ['sex', 'fasting_blood_sugar', 'exercise_induced_angina']
CAT = ['chest', 'resting_electrocardiographic_results', 'slope', 'thal']


def load(args=None):
    from sklearn.datasets import fetch_openml
    d = fetch_openml('heart-statlog', version=1, as_frame=True)
    df, target = d.data, d.target
    cols, names = [], []
    for c in CONT:
        cols.append(df[c].to_numpy(float)); names.append(c)
    for c in BIN:
        cols.append((df[c].to_numpy(float) > 0).astype(float)); names.append(c)
    for c in CAT:
        for lv in sorted(df[c].unique()):
            cols.append((df[c] == lv).to_numpy(float)); names.append(f'{c}={lv}')
    y = np.where(target.to_numpy() == 'present', 1, -1).astype(int)
    return np.column_stack(cols), np.asarray(names), y
