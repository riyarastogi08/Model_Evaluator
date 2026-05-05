from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime


class EvaluationCreate(BaseModel):
    model_name: str
    dataset_name: str
    task_type: str
    target_column: Optional[str] = None


class EvaluationResponse(EvaluationCreate):
    id: int
    status: str
    target_column: Optional[str] = None
    sensitive_attr: Optional[str] = None
    robustness_score: Optional[float] = None
    fairness_score: Optional[float] = None
    robustness_details: Optional[Dict[str, Any]] = None
    fairness_details: Optional[Dict[str, Any]] = None
    explainability_details: Optional[Dict[str, Any]] = None
    data_drift_details: Optional[Dict[str, Any]] = None
    rice_priority_table: Optional[List[Dict[str, Any]]] = None
    duration_seconds: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class EvaluationListResponse(BaseModel):
    """Lightweight schema for listing evaluations without full detail payloads."""
    id: int
    model_name: str
    dataset_name: str
    task_type: str
    status: str
    target_column: Optional[str] = None
    robustness_score: Optional[float] = None
    fairness_score: Optional[float] = None
    duration_seconds: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ComparisonResponse(BaseModel):
    """Side-by-side comparison of multiple evaluations."""
    evaluations: List[EvaluationResponse]
