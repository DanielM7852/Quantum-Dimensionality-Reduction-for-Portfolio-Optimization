from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, examples=["NVDA"])
    start_date: str = Field(..., examples=["2019-01-01"])
    end_date: str = Field(..., examples=["2024-11-01"])
    target_days: int = Field(default=5, ge=1, le=20)
    n_components: int = Field(default=8, ge=2, le=20)
    cv_splits: int = Field(default=5, ge=2, le=10)
    transaction_cost_bps: float = Field(default=10.0, ge=0, le=100)
    seed: int = Field(default=42, ge=0)
    entangledr_correlation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    entangledr_iterations: int = Field(default=3, ge=1, le=20)
    entangledr_rotation_scale: float = Field(default=0.08726646259971647, ge=0.001, le=1.0)

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        ticker = v.strip().upper()
        if not ticker.isalpha():
            raise ValueError("Ticker must contain letters only")
        return ticker


class ComboResult(BaseModel):
    dr_method: str
    classifier: str
    combo_id: str
    train_accuracy: float
    test_accuracy: float
    overfit_gap: float
    f1: float
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    research_grade: str
    accuracy_ci_lower: float | None = None
    accuracy_ci_upper: float | None = None
    sharpe_ci_lower: float | None = None
    sharpe_ci_upper: float | None = None
    return_ci_lower: float | None = None
    return_ci_upper: float | None = None
    brier_score: float | None = None
    deflated_sharpe: float | None = None


class BenchmarkResult(BaseModel):
    name: str
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    trade_count: int


class SignificanceResult(BaseModel):
    method_a: str
    method_b: str
    metric: str
    p_value: float
    significant: bool


class EquityPoint(BaseModel):
    date: str
    equity: float


class DRSummary(BaseModel):
    accuracy: float
    f1: float
    sharpe_ratio: float
    total_return: float
    sharpe_ci_lower: float | None = None
    sharpe_ci_upper: float | None = None


class AnalyzeMeta(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    sample_count: int
    positive_class_pct: float
    disclaimer_version: str
    disclaimer_text: str
    metrics_convention: str
    product_name: str = "EntangleDR Lab"
    seed: int = 42


class AnalyzeResponse(BaseModel):
    meta: AnalyzeMeta
    combos: list[ComboResult]
    best_combo_id: str | None
    equity_curves: dict[str, list[EquityPoint]]
    summary_by_dr: dict[str, DRSummary]
    benchmarks: list[BenchmarkResult] = []
    benchmark_curves: dict[str, list[EquityPoint]] = {}
    significance_tests: list[SignificanceResult] = []
