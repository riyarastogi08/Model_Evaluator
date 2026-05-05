from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.endpoints import router as eval_router
from app.core.config import settings
from app.core.database import engine
from app.core.logging_config import get_logger
from app.models.eval_report import Base

logger = get_logger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.APP_TITLE,
    description=(
        "API for evaluating ML models for robustness, fairness, drift, "
        "and explainability — prioritized with the RICE framework."
    ),
    version=settings.APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(eval_router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {
        "message": f"Welcome to the {settings.APP_TITLE} API",
        "version": settings.APP_VERSION,
    }


@app.get("/health")
def health_check():
    """
    Health-check endpoint for monitoring.
    Verifies DB connectivity so load balancers / uptime tools get an accurate signal.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:
        logger.error("Health check — DB unreachable: %s", exc)
        db_status = "unreachable"

    status = "healthy" if db_status == "ok" else "degraded"
    return {"status": status, "database": db_status, "version": settings.APP_VERSION}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
