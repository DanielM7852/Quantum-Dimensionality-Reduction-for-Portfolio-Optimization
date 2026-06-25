# Deploy EntangleDR Lab (portfolio-ready)

Get a public URL for your portfolio in about 15 minutes.

**Architecture:** Next.js on **Vercel** (frontend) + FastAPI on **Render** (backend API).

---

## Step 1: Push code to GitHub

From the repo root:

```bash
git add .
git commit -m "Prepare EntangleDR Lab for production deployment"
git push origin main
```

Repo: https://github.com/DanielM7852/Quantum-Dimensionality-Reduction-for-Portfolio-Optimization

---

## Step 2: Deploy backend (Render)

1. Go to [render.com](https://render.com) and sign in with GitHub.
2. **New** → **Blueprint** (or **Web Service**).
3. Connect this repository.
4. Render reads [`render.yaml`](../render.yaml) at the repo root:
   - Service name: `entangledr-lab-api`
   - Docker build from `backend/Dockerfile`
   - Health check: `/health`
5. After the first deploy, copy your API URL, e.g. `https://entangledr-lab-api.onrender.com`
6. In Render → **Environment**, set:
   ```
   CORS_ORIGINS=https://YOUR-VERCEL-URL.vercel.app,http://localhost:3000
   ```
   (Update again after Step 3 with your real Vercel URL.)

Free tier note: Render spins down after idle; first request may take 30-60s.

---

## Step 3: Deploy frontend (Vercel)

### Option A: Vercel dashboard (recommended)

1. Go to [vercel.com/new](https://vercel.com/new) and import the GitHub repo.
2. **Root Directory:** `frontend`
3. **Framework:** Next.js (auto-detected)
4. **Environment variable:**
   ```
   NEXT_PUBLIC_API_URL=https://entangledr-lab-api.onrender.com
   ```
   (Use your actual Render URL from Step 2.)
5. Deploy. Your site will be at `https://<project-name>.vercel.app`.

### Option B: Vercel CLI

```bash
cd frontend
vercel login
vercel link
vercel env add NEXT_PUBLIC_API_URL production
# paste: https://entangledr-lab-api.onrender.com
vercel --prod
```

---

## Step 4: Fix CORS (one-time)

1. Copy your final Vercel URL, e.g. `https://entangledr-lab.vercel.app`
2. In Render → `entangledr-lab-api` → **Environment**:
   ```
   CORS_ORIGINS=https://entangledr-lab.vercel.app,http://localhost:3000
   ```
3. **Save** and wait for redeploy.

---

## Step 5: Verify

1. Open `https://YOUR-VERCEL-URL.vercel.app`
2. Form should show **API connected** (green).
3. Run analysis on `NVDA` with default sliders.
4. Results should load in 30-90 seconds.

Backend health: `https://YOUR-RENDER-URL.onrender.com/health` → `{"status":"ok"}`

---

## Custom domain (optional)

**Vercel:** Project → **Settings** → **Domains** → add e.g. `entangledr.yourdomain.com`

**Render:** Service → **Settings** → **Custom Domains** (optional for API)

Update `CORS_ORIGINS` on Render if you add a custom frontend domain.

---

## Portfolio copy-paste

**Live demo:** `https://YOUR-VERCEL-URL.vercel.app`  
**GitHub:** `https://github.com/DanielM7852/Quantum-Dimensionality-Reduction-for-Portfolio-Optimization`  
**Stack:** Next.js, FastAPI, scikit-learn, Recharts, yfinance

**One-liner:** Interactive research lab comparing EntangleDR dimensionality reduction to PCA baselines on equity signals, with purged CV and live hyperparameter sliders.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| API offline on site | Check `NEXT_PUBLIC_API_URL` on Vercel; redeploy frontend |
| CORS error in browser | Add exact Vercel URL to `CORS_ORIGINS` on Render |
| Analysis timeout | Render cold start; retry or upgrade plan |
| 422 on ticker | Use letters only (e.g. `BRK-B` not supported) |
