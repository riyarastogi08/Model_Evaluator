"""
API endpoints for evaluation CRUD, status polling, history, and comparison.
"""
import os
import uuid
import shutil
from typing import List

from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Form, BackgroundTasks, Query, Request
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.logging_config import get_logger
from app.models.eval_report import EvaluationTask
from app.schemas.eval_report import EvaluationResponse, EvaluationListResponse, ComparisonResponse
from app.services.evaluator import run_evaluation_pipeline

logger = get_logger(__name__)
router = APIRouter()


# ─────────────────────────── helpers ───────────────────────────

def _secure_filename(original: str) -> str:
    """Prefix filename with a UUID to prevent overwrites and path traversal."""
    ext = os.path.splitext(original)[-1].lower()
    return f"{uuid.uuid4().hex}{ext}"


def _validate_extension(filename: str, allowed: list[str], label: str):
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {label} format '{ext}'. Allowed: {', '.join(allowed)}",
        )


async def _save_upload(upload: UploadFile, label: str, allowed: list[str], max_mb: int) -> str:
    """Validate and persist an uploaded file, returning the on-disk path."""
    _validate_extension(upload.filename, allowed, label)

    # File-size guard (read in chunks to avoid OOM on huge files)
    safe_name = _secure_filename(upload.filename)
    dest = os.path.join(settings.UPLOAD_DIR, safe_name)

    total_bytes = 0
    max_bytes = max_mb * 1024 * 1024
    with open(dest, "wb") as f:
        while chunk := await upload.read(1024 * 1024):  # 1 MB chunks
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                f.close()
                os.remove(dest)
                raise HTTPException(
                    status_code=413, detail=f"{label} exceeds {max_mb} MB limit."
                )
            f.write(chunk)

    return dest


# ─────────────────────────── POST /evaluate ───────────────────

@router.post("/evaluate/", response_model=EvaluationResponse)
async def create_evaluation_task(
    background_tasks: BackgroundTasks,          # ← injected by FastAPI (not constructed manually)
    model_name: str = Form(...),
    dataset_name: str = Form(...),
    task_type: str = Form(...),
    target_column: str = Form(""),              # ← user specifies the target column
    sensitive_attr: str = Form(""),
    model_file: UploadFile = File(...),
    dataset_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # Validate task_type
    if task_type.lower() not in ("classification", "regression"):
        raise HTTPException(
            status_code=400,
            detail="task_type must be 'classification' or 'regression'."
        )

    model_path = await _save_upload(model_file, "model", settings.ALLOWED_MODEL_EXTENSIONS, settings.MAX_FILE_SIZE_MB)
    dataset_path = await _save_upload(dataset_file, "dataset", settings.ALLOWED_DATASET_EXTENSIONS, settings.MAX_FILE_SIZE_MB)

    eval_task = EvaluationTask(
        model_name=model_name,
        dataset_name=dataset_name,
        task_type=task_type,
        target_column=target_column.strip() or None,
        sensitive_attr=sensitive_attr.strip() or None,
        status="pending",
    )
    db.add(eval_task)
    db.commit()
    db.refresh(eval_task)

    logger.info(
        "Created evaluation task %d (model=%s, dataset=%s, target=%s)",
        eval_task.id, model_name, dataset_name, target_column or "auto-last-col",
    )

    background_tasks.add_task(
        run_evaluation_pipeline,
        eval_task.id,
        model_path,
        dataset_path,
        task_type,
        target_column.strip() or "",
        sensitive_attr.strip(),
    )

    return eval_task


# ─────────────────────────── GET /status ──────────────────────

@router.get("/status/{task_id}", response_model=EvaluationResponse)
def get_evaluation_status(task_id: int, db: Session = Depends(get_db)):
    eval_task = db.query(EvaluationTask).filter(EvaluationTask.id == task_id).first()
    if not eval_task:
        raise HTTPException(status_code=404, detail="Task not found")
    return eval_task


# ─────────────────────────── GET /evaluations ─────────────────

@router.get("/evaluations/", response_model=List[EvaluationListResponse])
def list_evaluations(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List all past evaluations, newest first."""
    tasks = (
        db.query(EvaluationTask)
        .order_by(EvaluationTask.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return tasks


# ─────────────────────────── DELETE /evaluations/{id} ─────────

@router.delete("/evaluations/{task_id}")
def delete_evaluation(task_id: int, db: Session = Depends(get_db)):
    """Delete an evaluation record."""
    eval_task = db.query(EvaluationTask).filter(EvaluationTask.id == task_id).first()
    if not eval_task:
        raise HTTPException(status_code=404, detail="Task not found")

    db.delete(eval_task)
    db.commit()
    logger.info("Deleted evaluation task %d", task_id)
    return {"detail": f"Evaluation {task_id} deleted."}


# ─────────────────────────── GET /compare ─────────────────────

@router.get("/compare/", response_model=ComparisonResponse)
def compare_evaluations(
    ids: str = Query(..., description="Comma-separated evaluation IDs, e.g. 1,2,3"),
    db: Session = Depends(get_db),
):
    """
    Compare multiple evaluations side-by-side.
    Accepts a comma-separated string of evaluation IDs.
    """
    try:
        id_list = [int(i.strip()) for i in ids.split(",") if i.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="IDs must be comma-separated integers.")

    if len(id_list) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 evaluation IDs for comparison.")
    if len(id_list) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 evaluations can be compared at once.")

    tasks = db.query(EvaluationTask).filter(EvaluationTask.id.in_(id_list)).all()

    if len(tasks) != len(id_list):
        found = {t.id for t in tasks}
        missing = [i for i in id_list if i not in found]
        raise HTTPException(status_code=404, detail=f"Evaluations not found: {missing}")

    return ComparisonResponse(evaluations=tasks)
