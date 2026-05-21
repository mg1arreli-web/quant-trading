"""
Macro Regime Switch — HMM-based regime detection using VIX and TNX data.

Uses hmmlearn.hmm.GaussianHMM when available. Falls back to
sklearn.mixture.GaussianMixture when hmmlearn cannot be installed
(e.g., Python 3.14 where no pre-built wheel exists).
"""
import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Conditional HMM backend ────────────────────────────────────────────
try:
    from hmmlearn import hmm as _hmm_module
    HAS_HMMLEARN = True
    logger.debug("Using hmmlearn.hmm.GaussianHMM backend")
except ImportError:
    HAS_HMMLEARN = False
    logger.info("hmmlearn not available, falling back to sklearn GaussianMixture")


class _SklearnHMMFallback:
    """
    Drop-in replacement that uses sklearn GaussianMixture for regime detection.
    Provides the same .fit() / .predict() / .means_ interface as hmmlearn GaussianHMM.
    """
    def __init__(self, n_components: int = 2, random_state: int = 42):
        from sklearn.mixture import GaussianMixture
        self.n_components = n_components
        self._gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            max_iter=1000,
            random_state=random_state,
        )
        self.means_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        self._gmm.fit(X)
        self.means_ = self._gmm.means_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._gmm.predict(X)


class MacroRegimeSwitch:
    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        if HAS_HMMLEARN:
            self.model = _hmm_module.GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=1000,
                random_state=42,
            )
        else:
            self.model = _SklearnHMMFallback(n_components=n_regimes)
        self.data: pd.DataFrame | None = None

    def fetch_data(self) -> pd.DataFrame:
        try:
            vix = yf.download('^VIX', start='2010-01-01', end='2022-12-31')[['Close']]
            tnx = yf.download('^TNX', start='2010-01-01', end='2022-12-31')[['Close']]
        except Exception as e:
            raise ValueError(f"Failed to download from yfinance: {e}")

        vix.columns = ['VIX']
        tnx.columns = ['TNX']

        df = pd.concat([vix, tnx], axis=1).dropna()
        df['VIX_Ret'] = df['VIX'].pct_change()
        df['TNX_Ret'] = df['TNX'].pct_change()
        self.data = df.dropna()

        if self.data.empty:
            raise ValueError("No data available after processing VIX/TNX returns")

        logger.info(f"Fetched {len(self.data)} rows of VIX/TNX data for regime model")
        return self.data

    def fit(self, df: pd.DataFrame | None = None) -> None:
        self.fetch_data()
        self.features = self.data[['VIX_Ret', 'TNX_Ret']].values
        self.model.fit(self.features)
        logger.info(
            f"Regime model fitted with {self.n_regimes} regimes. "
            f"Means: {self.model.means_.tolist()}"
        )

    def predict_current_regime(self, recent_data: np.ndarray | None = None) -> int:
        recent_features = self.data[['VIX_Ret', 'TNX_Ret']].iloc[-1].values
        return int(self.model.predict(recent_features.reshape(1, -1))[0])

    def map_regime_to_risk(self, current_state: int) -> str:
        means = self.model.means_
        vix_means = means[:, 0]

        risk_off_state = int(np.argmax(vix_means))

        if current_state == risk_off_state:
            return "Risk-Off"
        else:
            return "Risk-On"
