from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from datetime import datetime, timezone
from app.core.database import Base


class EvaluationTask(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    dataset_name = Column(String)
    task_type = Column(String)         # 'classification' or 'regression'
    target_column = Column(String, nullable=True)   # user-specified target column
    sensitive_attr = Column(String, nullable=True)
    status = Column(String, default="pending")      # pending, evaluating, completed, failed

    # Scores
    robustness_score = Column(Float, nullable=True)
    fairness_score = Column(Float, nullable=True)

    # Detailed results
    robustness_details = Column(JSON, nullable=True)
    fairness_details = Column(JSON, nullable=True)
    explainability_details = Column(JSON, nullable=True)
    data_drift_details = Column(JSON, nullable=True)
    rice_priority_table = Column(JSON, nullable=True)

    # Meta
    duration_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
