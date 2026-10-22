import os
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field

API_KEY = os.getenv("INTEGRITY_API_KEY", "dev-api-key-2026")

class OvertimeScoreRequest(BaseModel):
    sn: str
    emp_code: str
    shift_date: str
    total_hours: float = Field(..., ge=0, le=24)
    overtime_hours: float = Field(..., ge=0, le=24)
    device_trust_state: str
    sensor_flags_count: int = Field(0, ge=0)

app = FastAPI()

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized API Key")
    return x_api_key

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ml/score-overtime")
def score(payload: OvertimeScoreRequest, api_key: str = Depends(verify_api_key)):
    # Task for Junior Dev: Implement Scikit-Learn logic here
    return {
        "sn": payload.sn,
        "emp_code": payload.emp_code,
        "shift_date": payload.shift_date,
        "total_hours": payload.total_hours,
        "overtime_hours": payload.overtime_hours,
        "anomaly_score": 0.0,
        "is_anomaly": False,
        "baseline_model": "RandomForest_Baseline",
        "explainability": {}
    }
