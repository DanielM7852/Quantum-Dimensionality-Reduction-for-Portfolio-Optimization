from datetime import date, datetime

from app.schemas import AnalyzeRequest

MIN_ANALYSIS_RANGE_DAYS = 365
MAX_ANALYSIS_RANGE_DAYS = 730
MAX_CV_SPLITS = 3
MAX_ENTANGLEDR_ITERATIONS = 3


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _days_between(start: str, end: str) -> int:
    return (_parse_date(end) - _parse_date(start)).days


def validate_for_hosted_api(request: AnalyzeRequest) -> None:
    """Reject web requests that are likely to exceed the hosted ~60s limit."""
    start = _parse_date(request.start_date)
    end = _parse_date(request.end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    span = _days_between(request.start_date, request.end_date)
    if span < MIN_ANALYSIS_RANGE_DAYS:
        raise ValueError(
            f"Date range must be at least {MIN_ANALYSIS_RANGE_DAYS} days (~1 year)."
        )
    if span > MAX_ANALYSIS_RANGE_DAYS:
        raise ValueError(
            f"Date range cannot exceed {MAX_ANALYSIS_RANGE_DAYS} days (~2 years) on the hosted API."
        )
    if request.cv_splits > MAX_CV_SPLITS:
        raise ValueError(f"cv_splits cannot exceed {MAX_CV_SPLITS} on the hosted API.")
    if request.entangledr_iterations > MAX_ENTANGLEDR_ITERATIONS:
        raise ValueError(
            f"entangledr_iterations cannot exceed {MAX_ENTANGLEDR_ITERATIONS} on the hosted API."
        )
