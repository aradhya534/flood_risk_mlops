from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def _sample_records(district="Ampara", zone="Dry", n_days=20):
    """20 days of synthetic history — same shape /predict expects."""
    records = []
    for i in range(n_days):
        records.append(
            {
                "date": f"2023-01-{i+1:02d}",
                "district": district,
                "climatic_zone": zone,
                "precipitation_sum": 5.0 + i,
                "rain_48h": 10.0,
                "rain_72h": 15.0,
                "soil_moisture_0_to_7cm_mean": 0.3,
                "soil_moisture_7_to_28cm_mean": 0.3,
                "soil_saturation_index": 0.3,
                "temperature_2m_max": 28.0,
                "wind_speed_10m_max": 10.0,
            }
        )
    return records


def test_predict_returns_200_for_valid_single_district():
    response = client.post("/predict", json={"records": _sample_records()})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert "flood_risk_score" in body[0]
    assert "advisory_predicted" in body[0]


def test_predict_rejects_multiple_districts():
    records = _sample_records("Ampara") + _sample_records("Kandy")
    response = client.post("/predict", json={"records": records})
    assert response.status_code == 422


def test_predict_batch_accepts_multiple_districts():
    records = _sample_records("Ampara") + _sample_records("Kandy")
    response = client.post("/predict/batch", json={"records": records})
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_predict_rejects_invalid_soil_moisture():
    records = _sample_records()
    records[0][
        "soil_moisture_0_to_7cm_mean"
    ] = 5.0  # out of the 0-1 range, should be rejected
    response = client.post("/predict", json={"records": records})
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    records = _sample_records()
    del records[0]["precipitation_sum"]
    response = client.post("/predict", json={"records": records})
    assert response.status_code == 422


def test_predict_rejects_empty_records():
    response = client.post("/predict", json={"records": []})
    assert response.status_code == 422
