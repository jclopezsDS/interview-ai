# Deployment Plan - InterviewAI

**Stack**: Railway (Backend) + Vercel (Frontend) + Sentry + UptimeRobot  
**Storage**: In-Memory Sessions  
**Strategy**: Academic Repo (Private) + Portfolio Repo (Public)  
**Timeline**: 3.5 hours

---

## 📦 Repository Strategy

### Two-Repo Approach (Turing College Guidelines Compliant)

```
┌─────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT WORKFLOW                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Academic Repo (PRIVATE)          Portfolio Repo (PUBLIC)   │
│  ┌─────────────────────┐          ┌─────────────────────┐  │
│  │ TuringCollege...    │          │ julopez/            │  │
│  │ julopez-AE.1.4      │          │ interview-ai        │  │
│  ├─────────────────────┤   COPY   ├─────────────────────┤  │
│  │ • backend/          │  ──────> │ • backend/          │  │
│  │ • frontend/         │ (clean)  │ • frontend/         │  │
│  │ • 114.md            │          │ • README.md (pro)   │  │
│  │ • DEPLOYMENT_PLAN   │  ──────> │ • DEPLOYMENT_PLAN   │  │
│  └─────────────────────┘          └──────────┬──────────┘  │
│           │                                   │             │
│           │                                   ├──> Railway  │
│           │                                   └──> Vercel   │
│           │                                                 │
│           └──> Turing College Review                        │
│               (stays private)                               │
└─────────────────────────────────────────────────────────────┘
```

**Repo 1: `TuringCollegeSubmissions/julopez-AE.1.4`** (PRIVATE)
- Purpose: Academic review only
- Contains: Assignment docs (`114.md`), evaluation criteria, project status
- Visibility: Private (stays private post-review)
- Deployment: NOT used for deployment

**Repo 2: `julopez/interview-ai`** (PUBLIC)
- Purpose: Portfolio + deployment
- Contains: Clean code, professional README, no academic references
- Visibility: Public (for LinkedIn, CV, recruiters)
- Deployment: Connected to Railway + Vercel

### File Distribution

| File/Folder | Academic Repo | Portfolio Repo |
|-------------|---------------|----------------|
| `backend/src/` | ✅ | ✅ |
| `backend/notebooks/` | ✅ | ✅ (cleaned) |
| `interview-app-frontend/` | ✅ | ✅ |
| `.gitignore` | ✅ | ✅ |
| `DEPLOYMENT_PLAN.md` | ✅ | ✅ (deployment context) |
| `114.md` | ✅ | ❌ |
| `README.md` | Academic version | Professional version |

### Setup Portfolio Repo

```bash
# Navigate to parent directory
cd /Users/julian/Desktop/TurinCollege/Module4/

# Create portfolio directory
mkdir interview-ai-portfolio
cd interview-ai-portfolio

# Copy production files only
cp -r ../julopez-AE.1.4/backend .
cp -r ../julopez-AE.1.4/interview-app-frontend .
cp ../julopez-AE.1.4/.gitignore .
cp ../julopez-AE.1.4/DEPLOYMENT_PLAN.md .

# Clean notebooks (remove academic references if needed)
# Review backend/notebooks/*.ipynb for Turing College mentions

# Initialize Git
git init
git add .
git commit -m "Initial commit: AI Interview Practice Platform"

# Create GitHub repo (public): interview-ai
# Then connect:
git remote add origin https://github.com/julopez/interview-ai.git
git push -u origin main

# Create production branch
git checkout -b production
git push origin production
```

### Professional README Template

Create `README.md` in portfolio repo:

```markdown
# InterviewAI - AI-Powered Interview Practice

> Full-stack application for AI-driven technical interview preparation

## Problem Solved
Candidates lack personalized, interactive practice for technical interviews. 
InterviewAI provides an AI interviewer that generates relevant questions, 
evaluates answers in real-time, and provides constructive feedback.

## Tech Stack
**Backend**: FastAPI, LangGraph, OpenAI GPT-4o-mini, Python 3.10  
**Frontend**: React 18, TypeScript, Vite, TailwindCSS, shadcn/ui  
**Deployment**: Railway (backend), Vercel (frontend)

## Features
- Dynamic question generation based on job description
- Real-time answer evaluation with detailed feedback
- Multiple interview types (Technical, Behavioral, Case Study)
- Configurable difficulty levels (Beginner, Intermediate, Advanced)
- 6-question structured interview flow with overall summary

## Architecture
- **Stateful AI Workflows**: LangGraph for conversation orchestration
- **Vector RAG**: Job description embedding for context-aware questions
- **Multi-LLM**: Explored GPT-4, Claude, Gemini for best performance
- **Security**: Input validation, rate limiting, CORS protection

## Live Demo
- **App**: https://interview-ai.vercel.app
- **API Docs**: https://interview-ai-backend.railway.app/docs

## Local Setup
[Standard setup instructions without academic references]

## Project Highlights
- Built stateful conversation system with LangGraph
- Implemented advanced prompt engineering techniques
- Optimized for cost (GPT-4o-mini) while maintaining quality
- Professional UX with real-time feedback and progress tracking

## Contact
Julian López - [LinkedIn](https://linkedin.com/in/yourprofile)
```

---

## 📋 Pre-Deployment Checklist

### Environment Configuration
- [ ] Create production `.env` with new OpenAI API key (separate from dev)
- [ ] Generate secure `SECRET_KEY` for FastAPI (if adding auth later)
- [ ] Document all environment variables in `.env.production.template`

### Code Preparation
- [ ] Remove all `console.log` and `print()` debug statements
- [ ] Set `FASTAPI_ENV=production` in backend
- [ ] Update CORS origins to production frontend URL
- [ ] Verify all API endpoints work locally with production settings
- [ ] Test full interview flow end-to-end locally

### Version Control
- [ ] **Academic repo**: Commit final version to `main`
- [ ] **Portfolio repo**: Already has `production` branch (from setup above)
- [ ] Tag portfolio release: `git tag v1.0.0-prod`
- [ ] Ensure `.env` files in `.gitignore` (both repos)

---


## 🚀 Backend Deployment (Railway)

**Working Directory**: `interview-ai-portfolio/backend/`

- [ ] Navigate: `cd /Users/julian/Desktop/TurinCollege/Module4/interview-ai-portfolio/backend`
- [ ] Create account: https://railway.app (no credit card required)
- [ ] Install CLI: `npm i -g railway`
- [ ] Login: `railway login` (OAuth with GitHub)
- [ ] Init project: `railway init`
- [ ] Select repo: `julopez/interview-ai` (public portfolio repo)
- [ ] Set environment variables:
  ```bash
  railway variables set OPENAI_API_KEY=<key>
  railway variables set FASTAPI_ENV=production
  railway variables set ALLOWED_ORIGINS=<frontend-url>
  ```
- [ ] Deploy: `railway up`
- [ ] Copy backend URL: `railway open`
- [ ] Test: `curl <railway-url>/health`

---

## 🎨 Frontend Deployment (Vercel)

**Working Directory**: `interview-ai-portfolio/interview-app-frontend/`

- [ ] Navigate: `cd ../interview-app-frontend`
- [ ] Create account: https://vercel.com
- [ ] Install CLI: `npm i -g vercel`
- [ ] Login: `vercel login` (OAuth with GitHub)
- [ ] Deploy: `vercel --prod`
- [ ] Select repo: `julopez/interview-ai` (public portfolio repo)
- [ ] Set env: `VITE_API_URL=<railway-backend-url>`
- [ ] Redeploy: `vercel --prod`
- [ ] Update Railway CORS with Vercel URL

---

## 🐳 Docker Configuration

- [ ] Create `backend/Dockerfile`:
  ```dockerfile
  FROM python:3.10-slim
  WORKDIR /app
  COPY pyproject.toml uv.lock ./
  RUN pip install uv && uv sync --frozen --no-dev
  COPY . .
  EXPOSE 8000
  CMD ["uv", "run", "uvicorn", "app_v2:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- [ ] Test locally: `docker build -t interview-backend . && docker run -p 8000:8000 --env-file .env interview-backend`
- [ ] (Optional) Push to Docker Hub: `docker push <username>/interview-backend:v1.0.0`

**Note**: Railway auto-detects and builds from Dockerfile.

---

## 🔐 Security Hardening

### Backend
- [ ] Restrict CORS to Vercel URL only
- [ ] Add rate limiting (optional):
  ```python
  pip install slowapi
  from slowapi import Limiter
  limiter = Limiter(key_func=get_remote_address)
  @app.post("/api/sessions")
  @limiter.limit("5/minute")
  ```
- [ ] Rotate OpenAI API key (production-only)

### Frontend
- [ ] Remove dev API URLs
- [ ] Verify no hardcoded secrets

---

## 📊 Monitoring & Logging

### Sentry (Error Tracking)
- [ ] Create account: https://sentry.io
- [ ] Create backend project (Python/FastAPI)
- [ ] Create frontend project (React)
- [ ] Backend: `pip install sentry-sdk[fastapi]`
- [ ] Add to `app_v2.py`:
  ```python
  import sentry_sdk
  sentry_sdk.init(
      dsn=os.getenv("SENTRY_DSN"),
      environment="production",
      traces_sample_rate=0.1
  )
  ```
- [ ] Frontend: `npm install @sentry/react`
- [ ] Configure in `main.tsx`
- [ ] Test with intentional error

### UptimeRobot (Uptime Monitoring)
- [ ] Create account: https://uptimerobot.com
- [ ] Add monitor: backend `/health` (5 min interval)
- [ ] Add monitor: frontend
- [ ] Configure email alerts
- [ ] Test by stopping backend

### Application Logs
- Railway dashboard has built-in logs
- (Optional) Add JSON logging:
  ```python
  import logging
  logging.basicConfig(
      level=logging.INFO,
      format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}'
  )
  ```

---

## 💰 Cost Management

### Estimated Monthly Costs
| Service | Free Tier | Notes |
|---------|-----------|-------|
| Railway | $5 credit | ~500 hrs/month free |
| Vercel | Unlimited | Free |
| OpenAI | Pay-per-use | ~$2-5/mo (light usage) |
| Sentry | 5K events/mo | Free |
| UptimeRobot | 50 monitors | Free |

**Total**: $2-5/month (OpenAI only)

### Cost Optimization
- [ ] Set OpenAI usage alert ($10 threshold)
- [ ] Monitor Sentry event count (<5K/mo)
- [ ] Use Railway sleep mode when not demoing

### Shutdown Strategy
```bash
railway down  # Stops backend, prevents costs
# Vercel stays up (free)
```

---

## ✅ Testing & Validation

### Pre-Launch Testing
- [ ] Full interview flow (6 questions + closing)
- [ ] Mobile responsiveness (iPhone, Android)
- [ ] Error scenarios:
  - [ ] Invalid API key → graceful error
  - [ ] Character limit exceeded → blocked with message
  - [ ] Network timeout → retry option
  - [ ] Session not found → redirect to home
- [ ] Performance:
  - [ ] Lighthouse score >90 for frontend
  - [ ] Backend response <3s for question generation
- [ ] Cross-browser: Chrome, Firefox, Safari
- [ ] CORS working (no console errors)

### Post-Deployment Validation
- [ ] `curl <backend-url>/health` returns 200
- [ ] API docs: `<backend-url>/docs`
- [ ] Frontend loads: `<frontend-url>`
- [ ] Full interview flow works
- [ ] Sentry receives test error
- [ ] UptimeRobot shows "Up"
- [ ] Check Railway logs

---

## 📸 Documentation & Portfolio

### Screenshots to Capture
- [ ] Landing page (interview configuration)
- [ ] Active interview (with progress indicator)
- [ ] Feedback message example
- [ ] Completion banner with overall summary
- [ ] Mobile view
- [ ] Backend API docs (`/docs`)

### GitHub Repository Polish
- [ ] Professional README with live demo links
- [ ] Add screenshots to `docs/screenshots/` folder
- [ ] Update badges with deployment status
- [ ] Add CHANGELOG.md (optional)
- [ ] Ensure all sensitive data removed (audit git history)

---

## 🔄 CI/CD (Manual Dispatch - Recommended for MVP)

- [ ] Create `.github/workflows/deploy.yml`:
  ```yaml
  name: Deploy to Production
  on: workflow_dispatch  # Manual trigger only
  jobs:
    deploy-backend:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - name: Deploy to Railway
          run: |
            npm i -g railway
            railway login --token ${{ secrets.RAILWAY_TOKEN }}
            railway up
    deploy-frontend:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - name: Deploy to Vercel
          run: |
            npm i -g vercel
            vercel --token ${{ secrets.VERCEL_TOKEN }} --prod
  ```
- [ ] Add GitHub secrets: `RAILWAY_TOKEN`, `VERCEL_TOKEN`

**Why Manual**: More control, prevents accidental deploys, good for demos.

---

## 🚨 Rollback Strategy

### Common Issues
1. **Backend fails**: Check Railway logs, verify env vars, test locally
2. **Frontend fails**: Check Vercel logs, verify `VITE_API_URL`
3. **CORS errors**: Update Railway `ALLOWED_ORIGINS`, redeploy
4. **OpenAI errors**: Verify API key, check billing

### Rollback
- Backend: `railway rollback` or redeploy previous commit
- Frontend: Vercel dashboard → Deployments → Promote previous

### Emergency Shutdown
```bash
railway down  # Stops backend, prevents OpenAI charges
```

---

## 📅 Deployment Timeline

### Session 1: Repository Setup (45 min)
- [ ] Clean code in academic repo (remove debug logs)
- [ ] Create portfolio repo (GitHub UI)
- [ ] Copy files to `interview-ai-portfolio/`
- [ ] Create professional README
- [ ] Initialize Git and push to `julopez/interview-ai`
- [ ] Create `production` branch

### Session 2: Pre-Deployment (30 min)
- [ ] Create production `.env` with new OpenAI key
- [ ] Test locally with production settings
- [ ] Verify all endpoints work
- [ ] Full interview flow test

### Session 3: Deploy (45 min)
- [ ] Railway backend setup (from portfolio repo)
- [ ] Vercel frontend setup (from portfolio repo)
- [ ] Configure CORS
- [ ] End-to-end production test

### Session 4: Monitoring (30 min)
- [ ] Sentry setup (backend + frontend)
- [ ] UptimeRobot setup
- [ ] Test error reporting

### Session 5: Documentation (30 min)
- [ ] Capture screenshots
- [ ] Update portfolio README with live URLs
- [ ] Final polish

**Total**: 3.25 hours

---

## ✅ Final Checklist Before Going Live

### Critical
- [ ] Backend `/health` returns 200
- [ ] Frontend loads without console errors
- [ ] Full interview flow works end-to-end
- [ ] No sensitive data (API keys, secrets) in code/logs
- [ ] CORS configured correctly
- [ ] Error tracking (Sentry) receiving events
- [ ] Uptime monitoring (UptimeRobot) configured

### Nice to Have
- [ ] Rate limiting enabled
- [ ] Docker build successful
- [ ] CI/CD pipeline working
- [ ] Screenshots captured
- [ ] README updated with live URLs
- [ ] Mobile tested

### Post-Launch
- [ ] Monitor Sentry for 24h (check for unexpected errors)
- [ ] Check OpenAI usage dashboard (ensure no cost spikes)
- [ ] Test from different devices/networks
- [ ] Share with peer reviewers
- [ ] Document any issues in GitHub Issues

---

## 🎓 Deliverables for Academic Review

### Required
1. ✅ Live backend URL (API accessible)
2. ✅ Live frontend URL (app accessible)
3. ✅ GitHub repository (public, clean history)
4. ✅ README with setup instructions
5. ✅ Screenshots

### Recommended
6. ✅ API documentation (`/docs` endpoint)
7. ✅ Deployment architecture diagram
8. ✅ Cost analysis document
9. ✅ Security considerations documented
10. ✅ Lessons learned / reflection

---

## 🔗 Quick Reference

### Platforms
- Railway: https://railway.app
- Vercel: https://vercel.com
- Sentry: https://sentry.io
- UptimeRobot: https://uptimerobot.com

### Docs
- Railway: https://docs.railway.app
- Vercel: https://vercel.com/docs
- FastAPI: https://fastapi.tiangolo.com/deployment/

---

## 📝 Notes

### Repository Workflow
- **Academic repo** (`TuringCollegeSubmissions/julopez-AE.1.4`): 
  - Submit for Turing College review
  - Keep private permanently
  - Contains all assignment docs
  
- **Portfolio repo** (`julopez/interview-ai`):
  - Deploy to Railway + Vercel
  - Share on LinkedIn/CV
  - Professional presentation (no academic references)

### Turing College Compliance
✅ No assignment instructions in public repo  
✅ No evaluation criteria exposed  
✅ No sprint names or Turing College mentions  
✅ Professional README (problem-focused, not assignment-focused)  
✅ Academic repo stays private post-review

### Deployment Scope
- **Demo/MVP**: Use in-memory sessions, skip rate limiting
- **Portfolio**: Add screenshots, video, polish README
- **Production** (future): Add Redis, rate limiting, backups
- **Reversible**: `railway down` stops everything, no long-term commitment
