"""Adversarial validation — QuantGPT
Copyright (c) 2026 Miasyster. Licensed under the MIT License.
https://github.com/Miasyster/QuantGPT

Adversarial validation layer for factor backtesting.
Complements anti_overfit.py with 4 destructive tests:
1. Label Permutation — shuffle forward returns, factor should lose significance
2. Temporal Shuffle — block shuffle to break time-series structure
3. Random Universe — random stock subset, factor should not generalize
4. Noise Injection — Gaussian noise on factor values, measure IC decay rate
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class AdvTestResult:
    name: str
    passed: bool
    details: dict


@dataclass
class AdversarialResult:
    score: float  # 0-100
    recommendation: str
    tests: list[AdvTestResult] = field(default_factory=list)
    passed_count: int = 0
    total_count: int = 4


def _daily_spearman_ic(df: pd.DataFrame, factor_col: str = "factor_value",
                       ret_col: str = "fwd_ret") -> pd.Series:
    valid = df.dropna(subset=[factor_col, ret_col])
    if valid.empty:
        return pd.Series(dtype=float)

    def _sp(g):
        if len(g) < 5 or g[factor_col].nunique() < 2:
            return np.nan
        corr, _ = sp_stats.spearmanr(g[factor_col], g[ret_col])
        return corr if not np.isnan(corr) else 0.0

    return valid.groupby("trade_date", group_keys=False).apply(_sp).dropna()


class AdversarialValidator:

    def __init__(self, factor_df: pd.DataFrame, holding_period: int = 5):
        self.df = factor_df.copy()
        self.df["trade_date"] = pd.to_datetime(self.df["trade_date"])
        self.holding_period = holding_period
        self._prepare_forward_returns()

    def _prepare_forward_returns(self):
        self.df = self.df.sort_values(["stock_code", "trade_date"])
        self.df["fwd_ret"] = (
            self.df.groupby("stock_code")["daily_ret"]
            .transform(
                lambda s: s.shift(-1)
                .rolling(self.holding_period, min_periods=self.holding_period)
                .sum()
                .shift(-(self.holding_period - 1))
            )
        )

    def run_all(self) -> AdversarialResult:
        tests = [
            self.test_label_permutation(),
            self.test_temporal_shuffle(),
            self.test_random_universe(),
            self.test_noise_injection(),
        ]
        passed = sum(1 for t in tests if t.passed)
        score = passed / 4 * 100

        if score >= 80:
            rec = "通过"
        elif score >= 60:
            rec = "基本通过"
        elif score >= 40:
            rec = "存疑"
        else:
            rec = "高风险"

        return AdversarialResult(
            score=score,
            recommendation=rec,
            tests=tests,
            passed_count=passed,
            total_count=4,
        )

    def test_label_permutation(self, n_perms: int = 50) -> AdvTestResult:
        """Shuffle forward returns across stocks within each date.

        If the factor still shows significant IC after shuffling labels,
        it's likely overfitted to noise patterns rather than real signal.

        Pass: real |IC| > 95th percentile of permuted |IC|s.
        """
        real_ic_series = _daily_spearman_ic(self.df)
        if len(real_ic_series) < 20:
            return AdvTestResult("标签置换检验", False, {"error": "IC数据不足"})

        real_ic = float(real_ic_series.mean())
        rng = np.random.RandomState(42)

        valid = self.df.dropna(subset=["factor_value", "fwd_ret"])
        sampled_dates = sorted(valid["trade_date"].unique())[::3]
        valid_sampled = valid[valid["trade_date"].isin(sampled_dates)]

        perm_ics = []
        for _ in range(n_perms):
            shuffled = valid_sampled.copy()
            shuffled["fwd_ret"] = shuffled.groupby("trade_date")["fwd_ret"].transform(
                lambda s: rng.permutation(s.values)
            )
            pic = _daily_spearman_ic(shuffled)
            if len(pic) > 0:
                perm_ics.append(float(pic.mean()))

        if len(perm_ics) < 10:
            return AdvTestResult("标签置换检验", False, {"error": "置换数据不足"})

        perm_95 = float(np.percentile([abs(x) for x in perm_ics], 95))
        passed = abs(real_ic) > perm_95

        return AdvTestResult("标签置换检验", passed, {
            "real_ic": round(real_ic, 4),
            "perm_95th_abs": round(perm_95, 4),
            "perm_mean_abs": round(float(np.mean([abs(x) for x in perm_ics])), 4),
            "n_perms": len(perm_ics),
        })

    def test_temporal_shuffle(self, block_size: int = 20) -> AdvTestResult:
        """Block shuffle the time series to break temporal structure.

        Divides the time series into blocks of `block_size` days,
        shuffles block order, then recalculates IC.

        Pass: real |IC| significantly > shuffled |IC| (ratio > 1.5).
        """
        real_ic_series = _daily_spearman_ic(self.df)
        if len(real_ic_series) < 40:
            return AdvTestResult("时序打乱检验", False, {"error": "数据不足"})

        real_ic = abs(float(real_ic_series.mean()))
        if real_ic < 1e-6:
            return AdvTestResult("时序打乱检验", False, {"error": "原始IC接近零"})

        dates = sorted(self.df["trade_date"].unique())
        n_blocks = len(dates) // block_size
        if n_blocks < 3:
            return AdvTestResult("时序打乱检验", False, {"error": "时间跨度不足以分块"})

        rng = np.random.RandomState(42)
        shuffle_ics = []

        for _ in range(30):
            blocks = [dates[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
            rng.shuffle(blocks)
            new_date_order = [d for block in blocks for d in block]
            date_map = dict(zip(dates[:len(new_date_order)], new_date_order))

            shuffled = self.df[self.df["trade_date"].isin(date_map.keys())].copy()
            shuffled["trade_date"] = shuffled["trade_date"].map(date_map)
            shuffled = shuffled.sort_values(["stock_code", "trade_date"])

            shuffled["fwd_ret"] = (
                shuffled.groupby("stock_code")["daily_ret"]
                .transform(
                    lambda s: s.shift(-1)
                    .rolling(self.holding_period, min_periods=self.holding_period)
                    .sum()
                    .shift(-(self.holding_period - 1))
                )
            )

            sic = _daily_spearman_ic(shuffled)
            if len(sic) > 0:
                shuffle_ics.append(abs(float(sic.mean())))

        if len(shuffle_ics) < 10:
            return AdvTestResult("时序打乱检验", False, {"error": "打乱数据不足"})

        mean_shuffled_ic = float(np.mean(shuffle_ics))
        ratio = real_ic / mean_shuffled_ic if mean_shuffled_ic > 1e-6 else 999.0
        passed = ratio > 1.5

        return AdvTestResult("时序打乱检验", passed, {
            "real_ic_abs": round(real_ic, 4),
            "shuffled_ic_abs_mean": round(mean_shuffled_ic, 4),
            "ratio": round(ratio, 2),
            "n_shuffles": len(shuffle_ics),
        })

    def test_random_universe(self, n_trials: int = 30, sample_frac: float = 0.3) -> AdvTestResult:
        """Test factor on random stock subsets.

        Randomly sample 30% of stocks, compute IC on each subset.
        A robust factor should maintain consistent IC sign across subsets.

        Pass: >= 70% of random subsets have IC same sign as full universe.
        """
        real_ic_series = _daily_spearman_ic(self.df)
        if len(real_ic_series) < 20:
            return AdvTestResult("随机股票池检验", False, {"error": "IC数据不足"})

        real_ic = float(real_ic_series.mean())
        real_sign = np.sign(real_ic)
        if real_sign == 0:
            return AdvTestResult("随机股票池检验", False, {"error": "原始IC为零"})

        stocks = self.df["stock_code"].unique()
        n_sample = max(10, int(len(stocks) * sample_frac))

        if len(stocks) < 20:
            return AdvTestResult("随机股票池检验", False, {"error": "股票数量不足"})

        rng = np.random.RandomState(42)
        subset_ics = []

        for _ in range(n_trials):
            sampled_stocks = rng.choice(stocks, size=n_sample, replace=False)
            subset = self.df[self.df["stock_code"].isin(sampled_stocks)]
            sic = _daily_spearman_ic(subset)
            if len(sic) >= 10:
                subset_ics.append(float(sic.mean()))

        if len(subset_ics) < 10:
            return AdvTestResult("随机股票池检验", False, {"error": "子集数据不足"})

        same_sign = sum(1 for ic in subset_ics if np.sign(ic) == real_sign)
        consistency = same_sign / len(subset_ics)
        passed = consistency >= 0.7

        return AdvTestResult("随机股票池检验", passed, {
            "real_ic": round(real_ic, 4),
            "consistency": round(consistency, 4),
            "subset_ic_mean": round(float(np.mean(subset_ics)), 4),
            "subset_ic_std": round(float(np.std(subset_ics)), 4),
            "n_trials": len(subset_ics),
        })

    def test_noise_injection(self, noise_levels: list[float] | None = None) -> AdvTestResult:
        """Add Gaussian noise to factor values and measure IC decay.

        Noise levels are multiples of factor_value std.
        A robust factor should degrade gracefully, not collapse.

        Pass: IC at noise_level=0.5 retains >= 50% of original |IC|.
        """
        if noise_levels is None:
            noise_levels = [0.1, 0.2, 0.5, 1.0, 2.0]

        real_ic_series = _daily_spearman_ic(self.df)
        if len(real_ic_series) < 20:
            return AdvTestResult("噪声注入检验", False, {"error": "IC数据不足"})

        real_ic = abs(float(real_ic_series.mean()))
        if real_ic < 1e-6:
            return AdvTestResult("噪声注入检验", False, {"error": "原始IC接近零"})

        rng = np.random.RandomState(42)
        factor_std = self.df["factor_value"].std()
        if factor_std < 1e-10:
            return AdvTestResult("噪声注入检验", False, {"error": "因子值无变异"})

        noise_ics = {}
        for level in noise_levels:
            noisy = self.df.copy()
            noise = rng.normal(0, factor_std * level, size=len(noisy))
            noisy["factor_value"] = noisy["factor_value"] + noise
            nic = _daily_spearman_ic(noisy)
            if len(nic) > 0:
                noise_ics[level] = abs(float(nic.mean()))

        if 0.5 not in noise_ics:
            return AdvTestResult("噪声注入检验", False, {"error": "0.5倍噪声计算失败"})

        retention_50 = noise_ics[0.5] / real_ic
        passed = retention_50 >= 0.5

        decay_curve = {str(k): round(v, 4) for k, v in sorted(noise_ics.items())}

        return AdvTestResult("噪声注入检验", passed, {
            "real_ic_abs": round(real_ic, 4),
            "noise_ics": decay_curve,
            "retention_at_0.5x": round(retention_50, 4),
            "factor_std": round(float(factor_std), 6),
        })


def run_adversarial_validation(factor_df: pd.DataFrame, holding_period: int = 5) -> dict:
    validator = AdversarialValidator(factor_df, holding_period)
    result = validator.run_all()
    return {
        "score": result.score,
        "recommendation": result.recommendation,
        "passed_count": result.passed_count,
        "total_count": result.total_count,
        "tests": [
            {
                "name": t.name,
                "passed": t.passed,
                "details": t.details,
            }
            for t in result.tests
        ],
    }
