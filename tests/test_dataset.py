"""Tests de dataset.py : exécuter avec pytest test_dataset.py."""
import numpy as np
import pandas as pd
import pytest

from moobs.dataset import build_dataset, common_test_index, usable_rows


@pytest.fixture(scope="module")
def forecast_df():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-02", periods=105, freq="7D")
    frames = []
    for store_num in range(1, 139):
        sales = rng.integers(100, 160, size=len(dates))
        store_df = pd.DataFrame({
            "store_id": f"S{store_num:03d}",
            "date": dates,
            "sales": sales,
        })
        store_df["sales_t"] = store_df["sales"]
        store_df["target_date"] = store_df["date"].shift(-2)
        store_df["target_sales"] = store_df["sales"].shift(-2)
        for lag in (1, 2, 4, 52):
            store_df[f"lag_{lag}"] = store_df["sales"].shift(lag)
        frames.append(store_df)
    return pd.concat(frames, ignore_index=True)


A = ["sales_t", "lag_1", "lag_2", "lag_4"]
B = A + ["lag_52"]
CUT = "2024-07-01"


def test_target_not_in_X(forecast_df):
    with pytest.raises(ValueError, match="cible.*variables explicatives"):
        build_dataset(forecast_df, A + ["target_sales"], CUT)


def test_no_nan_and_temporal_split(forecast_df):
    split = build_dataset(forecast_df, A, CUT)
    assert not split.X_train.isna().any().any() and not split.y_train.isna().any()
    assert not split.X_test.isna().any().any() and not split.y_test.isna().any()
    assert split.train_df["target_date"].max() < pd.Timestamp(CUT)
    assert split.test_df["target_date"].min() >= pd.Timestamp(CUT)
    assert list(split.X_train.columns) == A
    assert "target_sales" not in split.X_test.columns


def test_row_counts_match_expected_lag_loss(forecast_df):
    n = usable_rows(forecast_df, A).sum()
    assert n == 138 * (105 - 4 - 2)


def test_usable_rows_detects_missing_values():
    df = pd.DataFrame({
        "feature": [10, None],
        "target_sales": [100, 200],
        "target_date": ["2024-01-01", "2024-01-08"]
    })

    result = usable_rows(df, ["feature"])

    assert result.tolist() == [True, False]
    
    

def test_same_test_set_across_models(forecast_df):
    idx = common_test_index(forecast_df, [A, B], CUT)
    split_a = build_dataset(forecast_df, A, CUT, test_index=idx)
    split_b = build_dataset(forecast_df, B, CUT, test_index=idx)
    assert split_a.X_test.index.equals(split_b.X_test.index)
    assert split_a.y_test.equals(split_b.y_test)
    assert len(split_b.X_train) < len(split_a.X_train)


def test_imposed_test_rows_must_be_usable(forecast_df):
    # Ici, le cutoff est volontairement précoce : le test A contient des lignes
    # dont lag_52 est manquant. Ces lignes doivent être refusées pour B.
    early_cut = "2023-06-01"
    idx = common_test_index(forecast_df, [A], early_cut)
    with pytest.raises(ValueError, match="ne sont pas utilisables"):
        build_dataset(forecast_df, B, early_cut, test_index=idx)
