import hashlib
import json
import time
from collections import OrderedDict

from fastapi import APIRouter, HTTPException, Request

from app.core.analysis_limits import validate_for_hosted_api
from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.pipeline import run_analysis

router = APIRouter(prefix="/api/v1", tags=["analysis"])

_cache: OrderedDict[str, tuple[float, AnalyzeResponse]] = OrderedDict()
CACHE_TTL_SECONDS = 3600
CACHE_MAX_SIZE = 32

_rate_limit: dict[str, list[float]] = {}
RATE_LIMIT_MAX = 5
RATE_LIMIT_WINDOW = 60


def _cache_key(request: AnalyzeRequest) -> str:
    payload = request.model_dump()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _get_cached(key: str) -> AnalyzeResponse | None:
    if key not in _cache:
        return None
    ts, response = _cache[key]
    if time.time() - ts > CACHE_TTL_SECONDS:
        del _cache[key]
        return None
    _cache.move_to_end(key)
    return response


def _set_cache(key: str, response: AnalyzeResponse) -> None:
    _cache[key] = (time.time(), response)
    if len(_cache) > CACHE_MAX_SIZE:
        _cache.popitem(last=False)


def _check_rate_limit(client_ip: str) -> None:
    now = time.time()
    hits = _rate_limit.get(client_ip, [])
    hits = [t for t in hits if now - t < RATE_LIMIT_WINDOW]
    if len(hits) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded (5 requests per minute)")
    hits.append(now)
    _rate_limit[client_ip] = hits


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, req: Request):
    client_ip = req.client.host if req.client else "unknown"
    _check_rate_limit(client_ip)

    try:
        validate_for_hosted_api(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    key = _cache_key(request)
    cached = _get_cached(key)
    if cached:
        return cached

    try:
        result = run_analysis(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    _set_cache(key, result)
    return result
