import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def make_fwd_logret(rets: pd.DataFrame, horizon=12):
    return rets.rolling(horizon).sum().shift(-horizon)

def train_regime_conditional_alpha(X: pd.DataFrame, y: pd.DataFrame, regime_probs: pd.DataFrame,
                                  assets, x_cols, horizon=12, alpha=10.0, min_train=120):
    """
    For each regime k, fit Ridge: y_asset ~ X in that regime-weighted manner.
    Use sample weights = pRk (soft assignment).
    Produces expected returns each date via sum_k pRk * pred_k.
    """
    # align
    data = X.join(regime_probs, how="inner").join(y, how="inner").dropna(subset=x_cols)
    pcols = [c for c in regime_probs.columns if c.startswith("pR")]
    nR = len(pcols)

    models = {k: {a: Pipeline([("sc",StandardScaler()),("rg",Ridge(alpha=alpha))]) for a in assets} for k in range(nR)}
    exp = pd.DataFrame(index=data.index, columns=assets, dtype=float)

    # y at row j is the return over rows j+1 .. j+horizon, so it is only known from row j+horizon on.
    # Train row t's models on rows < t - horizon: every target they use is realised by row t.
    for t in range(min_train + horizon, len(data)):
        train = data.iloc[:t - horizon]
        test = data.iloc[t:t+1]

        # per regime per asset
        preds_k = np.zeros((nR, len(assets)))
        for k in range(nR):
            w_full = train[pcols[k]].values
            if np.nanmean(w_full) < 1e-3:
                continue
            for j,a in enumerate(assets):
                valid = ~np.isnan(train[a].values) & ~np.isnan(w_full)
                yy = train[a].values[valid]
                XX = train[x_cols].values[valid]
                w_valid = w_full[valid]

                if len(yy) < 10:
                    continue

                models[k][a].fit(XX, yy, rg__sample_weight=w_valid)
                preds_k[k, j] = models[k][a].predict(test[x_cols].values)[0]

        # mixture of experts
        pk = test[pcols].values[0]
        if not np.any(np.isnan(pk)):
            exp.iloc[t] = (pk.reshape(-1,1) * preds_k).sum(axis=0)

    return exp
