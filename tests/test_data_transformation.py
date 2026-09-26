import pandas as pd
import numpy as np

from src.components.data_transformation import add_input_features


def make_district_history(n_days: int = 20) -> pd.DataFrame:
    """
    A small, synthetic 30 day history for one district - enough to fill
     every rolling window (longest is 7 days) with room to spare
    """

    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "district": "TestDistrict",
            "climatic_zone": "Wet",
            "precipitation_sum": np.linspace(0, 50, n_days),
            "soil_moisture_0_to_7cm_mean": np.linspace(0.2, 0.6, n_days),
            "soil_moisture_7_to_28cm_mean": np.linspace(0.2, 0.5, n_days),
            "soil_saturation_index": np.linspace(0.2, 0.55, n_days),
            "temperature_2m_max": 28.0,
            "wind_speed_10m_max": 10.0,
            "rain_48h": 0.0,  # present in raw data, unused by add_input_features itself
            "rain_72h": 0.0,
        }
    )


def test_add_input_features_creates_expected_columns():
    df = add_input_features(make_district_history())
    for col in [
        "rain_lag1",
        "rain_lag2",
        "rain_lag3",
        "rain_3d",
        "rain_7d",
        "rain_max_7d",
        "wet_days_7d",
        "soil_change_3d",
        "soil_mean_7d",
        "month_sin",
        "month_cos",
    ]:
        assert col in df.columns


def test_rolling_features_need_enough_history():
    df = add_input_features(make_district_history(n_days=20))

    assert df["rain_7d"].iloc[:6].isna().all()
    assert df["rain_7d"].iloc[7:].notna().all()


def test_rain_7d_matches_manual_sum():
    df = add_input_features(make_district_history())
    manual_sum = df["precipitation_sum"].iloc[13:20].sum()
    assert np.isclose(df["rain_7d"].iloc[19], manual_sum)


def test_multiple_districts_dont_leak_into_each_other():
    d1 = make_district_history()
    d2 = make_district_history()
    d2["district"] = "OtherDistrict"
    d2["precipitation_sum"] = 999.0

    combined = pd.concat([d1, d2]).reset_index(drop=True)
    df = add_input_features(combined)

    d1_result = df[df["district"] == "TestDistrict"]

    assert d1_result["rain_7d"].max() < 999
