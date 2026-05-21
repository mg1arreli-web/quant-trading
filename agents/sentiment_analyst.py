"""
Sentiment Analyst — Residual Sentiment Momentum (RSM) factor.
"""
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def calculate_residual_sentiment_momentum(
    df: pd.DataFrame,
    sentiment_col: str = 'llm_sentiment',
    return_col: str = 'daily_return',
    reg_window: int = 21,
    mom_window: int = 5,
) -> pd.DataFrame:
    """
    Calculate the Residual Sentiment Momentum factor using rolling OLS regression.
    """
    df = df.copy()
    df['residual_sentiment'] = np.nan

    for i in range(reg_window, len(df)):
        y_train = df[sentiment_col].iloc[i - reg_window:i]
        X_train = sm.add_constant(df[return_col].iloc[i - reg_window:i])
        try:
            model = sm.OLS(y_train, X_train).fit()
            actual_sentiment = df[sentiment_col].iloc[i]
            ret_today = df[return_col].iloc[i]
            expected_sentiment = model.params.iloc[0] + (model.params.iloc[1] * ret_today)
            df.iloc[i, df.columns.get_loc('residual_sentiment')] = actual_sentiment - expected_sentiment
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.debug(f"OLS regression failed at index {i}: {e}")
            continue

    df['RSM_Factor'] = df['residual_sentiment'].rolling(window=mom_window).mean()
    return df
