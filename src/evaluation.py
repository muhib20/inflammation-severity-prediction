import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

def evaluate_regression(model, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    mae_scores = []

    for tr, te in gkf.split(X, y, groups=groups):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[te])
        mae_scores.append(mean_absolute_error(y.iloc[te], preds))

    return np.mean(mae_scores), np.std(mae_scores)
