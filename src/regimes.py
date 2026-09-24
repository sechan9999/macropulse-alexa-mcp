import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def fit_gmm_regimes(df_feat: pd.DataFrame, cols, n_regimes=3, min_train=120, order_by="credit_spread"):
    """
    Expanding fit of GMM; outputs regime probabilities each month.

    A GMM's component numbering is arbitrary and changes from one refit to the next, so pR0 in one
    month could be the stress regime and pR0 the next month the calm one. After every fit the
    components are sorted by their mean of `order_by` (default: credit spread), so pR0 is always
    the lowest-spread (calmest) regime and pR{n-1} the highest-spread one, and the probability
    columns mean the same thing in every row.
    """
    cols = list(cols)
    if order_by not in cols:
        order_by = cols[0]
    k_order = cols.index(order_by)

    X = df_feat[cols].dropna().copy()
    idx = X.index

    scaler = StandardScaler()
    probs = pd.DataFrame(index=idx, columns=[f"pR{i}" for i in range(n_regimes)], dtype=float)
    labels = pd.Series(index=idx, dtype=float)

    for t in range(min_train, len(X)):
        train = X.iloc[:t]
        test = X.iloc[t:t+1]

        Z = scaler.fit_transform(train.values)
        gmm = GaussianMixture(n_components=n_regimes, covariance_type="full", random_state=7)
        gmm.fit(Z)

        order = np.argsort(gmm.means_[:, k_order], kind="stable")   # calm → stressed
        zt = scaler.transform(test.values)
        pt = gmm.predict_proba(zt)[0][order]
        probs.iloc[t] = pt
        labels.iloc[t] = int(np.argmax(pt))

    out = probs.join(labels.rename("regime"))
    return out
