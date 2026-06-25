# Archived 2025-06-14: original monolithic script moved to archive/true_test_2025-06-14.py
"""Thin CLI wrapper for local pipeline debugging."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from app.schemas import AnalyzeRequest
from app.services.pipeline import run_analysis

CONFIG = {
    "ticker": "NVDA",
    "start_date": "2019-01-01",
    "end_date": "2024-11-01",
    "target_days": 5,
    "n_components": 8,
    "cv_splits": 5,
    "transaction_cost_bps": 10,
}


def main():
    request = AnalyzeRequest(**CONFIG)
    result = run_analysis(request)
    print(f"Ticker: {result.meta.ticker} | Samples: {result.meta.sample_count}")
    print(f"Best combo: {result.best_combo_id}")
    for combo in result.combos:
        print(
            f"  {combo.dr_method} + {combo.classifier}: "
            f"acc={combo.test_accuracy:.3f}, sharpe={combo.sharpe_ratio:.2f}, "
            f"return={combo.total_return:.2%}, grade={combo.research_grade}"
        )


if __name__ == "__main__":
    main()
