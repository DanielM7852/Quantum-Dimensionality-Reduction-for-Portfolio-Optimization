# EntangleDR Lab — Research Platform

A publishable research platform comparing **EntangleDR**, **PCA**, **Kernel-PCA**, and **No-DR** baselines with multiple classifiers on stock tickers. Purged time-series CV, bootstrap CIs, and significance testing. Educational tool — **not financial advice**.

## Product

**EntangleDR Lab** — *Purged cross-validation benchmark for correlation-aware dimensionality reduction in financial signal research.*

Algorithm spec: [`docs/method_entangledr.md`](docs/method_entangledr.md)  
Paper draft: [`docs/paper_entangledr_draft.md`](docs/paper_entangledr_draft.md)

## Completed Studies (June 2026)

| Study | Universe | Tickers | Type | Key result |
|-------|----------|---------|------|------------|
| `study_20260615_025409` | S&P 500 sample | 29 | Full 12-combo | EntangleDR mean Sharpe 0.656 vs PCA 0.656; wins 16/29 tickers |
| `study_20260616_125849` | Tech 10 | 10 | Full 12-combo | EntangleDR 0.931 vs PCA 0.925; wins 6/10 tickers |
| `study_20260615_030256` | Tech 10 | 10 | Ablation (27 configs) | All configs identical — saturation at d=9 features |

**Primary finding:** EntangleDR is statistically indistinguishable from PCA at d=9 features. All 27 ablation configs produce identical Sharpe per ticker. Next step: expand to d≥30 features.

## Architecture

```
Next.js (Vercel)  ──POST /api/v1/analyze──►  FastAPI (Render/Railway)
                                                    │
                                                    ▼
                                              yfinance data
```

| Component | Stack | Default host |
|-----------|-------|--------------|
| Frontend | Next.js 15, Tailwind, Recharts | Vercel |
| Backend | FastAPI, scikit-learn, yfinance | Render (Docker) |

## Disclaimer

This tool is for **educational and research purposes only**. It is not investment advice. Past simulated performance does not guarantee future results. Data from Yahoo Finance may be delayed or inaccurate. Do not use this to make real trading decisions.

## Local development

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check: http://localhost:8000/health

### Frontend

```bash
cd frontend
cp .env.example .env.local
# Edit NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev
```

Open http://localhost:3000

### CLI (legacy wrapper)

```bash
python true_test.py
```

### Batch experiments

```bash
cd backend
python -m app.experiments.run_study --universe sp500_sample --dry-run --limit 10
python -m app.experiments.run_study --universe sp500_sample --limit 10
python -m app.experiments.run_study --universe tech_10 --ablation --limit 5
```

Results written to `results/study_*.json` and `results/manifest.json`.

Original monolithic script archived at `archive/true_test_2025-06-14.py`.

## Deployment (portfolio)

Full guide: [`docs/DEPLOY.md`](docs/DEPLOY.md)

| Step | Platform | Result |
|------|----------|--------|
| 1 | GitHub | Source code |
| 2 | [Render](https://render.com) | API at `https://entangledr-lab-api.onrender.com` |
| 3 | [Vercel](https://vercel.com) | Site at `https://<your-project>.vercel.app` |

**Portfolio links (fill in after deploy):**
- Live demo: `https://YOUR-PROJECT.vercel.app`
- Repo: https://github.com/DanielM7852/Quantum-Dimensionality-Reduction-for-Portfolio-Optimization

### Backend (Render)

1. Push repo to GitHub
2. Create a **Web Service** on [Render](https://render.com) using `render.yaml` or:
   - Root directory: `backend`
   - Docker build from `backend/Dockerfile`
3. Set environment variable:
   ```
   CORS_ORIGINS=https://your-app.vercel.app,http://localhost:3000
   ```

### Frontend (Vercel)

1. Import repo on [Vercel](https://vercel.com)
2. Set **Root Directory** to `frontend`
3. Set environment variable:
   ```
   NEXT_PUBLIC_API_URL=https://your-api.onrender.com
   ```

## API

### `GET /health`

Returns `{ "status": "ok" }`.

### `POST /api/v1/analyze`

```json
{
  "ticker": "NVDA",
  "start_date": "2019-01-01",
  "end_date": "2024-11-01",
  "target_days": 5,
  "n_components": 8,
  "cv_splits": 5,
  "transaction_cost_bps": 10
}
```

Returns combo metrics, equity curves, and research grades. Rate limited to 5 requests/minute per IP. Results cached for 1 hour.

## Metrics convention

- **Sharpe ratio**: computed on per-trade returns from non-overlapping long-only entries, annualized with `sqrt(252/hold_days)`
- **Total return**: compound equity from non-overlapping trades after round-trip transaction costs
- **Research grade**: `promising`, `inconclusive`, or `overfit_risk` based on accuracy, overfit gap, and Sharpe

## Project structure

```
backend/app/          FastAPI + ML pipeline
frontend/             Next.js UI
archive/              Original research script
true_test.py          Thin CLI wrapper
```

## Validation checklist

- [x] NVDA run returns realistic returns (not 700%+ inflated sums)
- [x] Invalid ticker returns 422
- [ ] CORS works from Vercel URL (configure `CORS_ORIGINS` after deploy)
- [x] Disclaimer visible on load
- [x] API timeout shows friendly error in UI (120s client timeout)
- [x] Compare 2 tickers (NVDA vs SPY) — results differ sensibly
