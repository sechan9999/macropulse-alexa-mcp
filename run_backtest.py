import pandas as pd
from src.data_fred import load_fred
from src.data_prices import load_prices, log_returns
from src.features import add_macro_features, add_technical_features
from src.regimes import fit_gmm_regimes
from src.exp_return import make_fwd_logret, train_regime_conditional_alpha
from src.backtest import run_backtest
import os

def main():
    print("1. Loading Prices Data...")
    tickers = ["SPY","IEF","TLT","GLD","BIL"]
    px = load_prices(tickers, start="2006-01-01")
    rets = log_returns(px)

    print("2. Downloading FRED Macro Data (each value dated by its release, not its reference month)...")
    macro = load_fred(start="2000-01-01")
    macro = add_macro_features(macro)

    print("3. Generating technical features...")
    tech = add_technical_features(rets)

    # align features
    feat = macro.join(tech, how="inner").dropna()

    print("4. Classifying Market Regimes via GMM...")
    # regimes (pR0 = lowest-credit-spread regime ... pR2 = highest; stable across the monthly refits)
    reg_cols = ["real10y_proxy","credit_spread","yc_slope","cpi_yoy","indpro_yoy","nfci_z"]
    regimes = fit_gmm_regimes(feat, cols=reg_cols, n_regimes=3, min_train=120)

    # Make target format for dashboard rendering
    regimes.to_parquet("data/regimes.parquet")

    print("5. Training regime-conditional expected returns (Alpha)...")
    # forward returns target (12m) for training alpha
    y12 = make_fwd_logret(rets, horizon=12)
    y12.columns = tickers  # same names

    # X for alpha model (국면 + 시그널 기반)
    X_cols = ["real10y_proxy","credit_spread","yc_slope","cpi_yoy","indpro_yoy","nfci_z","mom_12_1_spy","rv_12_spy","dd_spy"]
    X = feat[X_cols].copy()

    exp12 = train_regime_conditional_alpha(
        X=X, y=y12, regime_probs=regimes[[c for c in regimes.columns if c.startswith("pR")]],
        assets=tickers, x_cols=X_cols, horizon=12, alpha=10.0, min_train=150
    )

    exp12.to_parquet("data/exp_return_estimates_raw.parquet")

    # convert 12m expected log-return to 1m expected log-return (simple approximation)
    exp1m = exp12 / 12.0

    print("6. Executing Walk-Forward Backtest (CVaR + turnover Constraints)...")
    # bounds (lo/hi)
    import numpy as np
    n = len(tickers)
    lo = np.array([0.00, 0.00, 0.00, 0.00, 0.00])  # long-only
    hi = np.array([0.80, 0.70, 0.50, 0.50, 0.50])
    bounds = (lo, hi)

    port, w_df = run_backtest(rets_log=rets[tickers], exp_log=exp1m[tickers],
                              lookback=60, start_idx=180,
                              lam=6.0, gamma=15.0, alpha=0.95,
                              bounds=bounds)

    # quick stats
    ann = (1+port).prod() ** (12/len(port)) - 1
    vol = port.std() * (12**0.5)
    excess = port - np.expm1(rets["BIL"]).reindex(port.index)   # in excess of T-bills (BIL ETF)
    sharpe = excess.mean() * 12 / (excess.std() * (12**0.5)) if excess.std() > 0 else float("nan")
    dd = (1+port).cumprod()
    mdd = (dd/dd.cummax()-1).min()

    print("\n================== BACKTEST RESULTS ==================")
    print(f"AnnReturn={ann:.2%}  Vol={vol:.2%}  Sharpe={sharpe:.2f}  MaxDD={mdd:.2%}")
    print("======================================================")
    
    port.to_csv("data/out_port_returns.csv")
    w_df.to_csv("data/out_weights.csv")
    macro.to_parquet("data/macro_equity_monthly.parquet")
    print("\nSaved artifacts to data/ directory: out_port_returns.csv, out_weights.csv, macro_equity_monthly.parquet, regimes.parquet")

if __name__ == "__main__":
    main()
