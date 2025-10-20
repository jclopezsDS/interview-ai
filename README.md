# InterviewAI - AI-Powered Interview Practice

> AI interviewer for technical interview preparation with real-time feedback. Practice technical, behavioral, and case study interviews with personalized questions and detailed evaluation.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![React](https://img.shields.io/badge/React-18.3-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-AI%20Orchestration-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Features

- **Dynamic Question Generation** - Tailored to job description and your background
- **Real-time Evaluation** - Instant feedback on strengths and areas to improve
- **Multiple Interview Types** - Technical, Behavioral, Case Study
- **Configurable Difficulty** - Beginner, Intermediate, Advanced
- **6-Question Flow** - Structured interview with overall summary
- **Professional UI** - Dark mode, responsive, mobile-friendly

---

## 🌐 Live Demo

**Frontend**: https://interview-app-frontend-mu.vercel.app/  
**Backend API**: https://interview-ai-production-7e11.up.railway.app/

---

## 🏗️ Tech Stack

**Backend**: FastAPI, LangGraph, OpenAI GPT-4o-mini, LangChain  
**Frontend**: React 18, TypeScript, Vite, TailwindCSS, shadcn/ui  
**Deployment**: Railway (backend), Vercel (frontend)

---

## 🚀 Quick Start

### Backend
```bash
cd backend
uv sync
cp .env.template .env  # Add OPENAI_API_KEY
uv run uvicorn app_v2:app --reload
```
→ http://localhost:8000

### Frontend
```bash
cd interview-app-frontend
npm install
npm run dev
```
→ http://localhost:5173

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/sessions` | Create interview session |
| `POST` | `/api/sessions/{id}/message` | Send user message |
| `GET` | `/api/sessions/{id}` | Get session state |
| `DELETE` | `/api/sessions/{id}` | Delete session |

**Docs**: http://localhost:8000/docs (when running)

---

## 🎨 Architecture

**LangGraph State Machine**: Greeting → Question Generation → Answer Evaluation → Feedback → (repeat 6x) → Summary

**Key Technologies**:
- LangGraph for stateful conversation workflows
- GPT-4o-mini for cost-effective AI responses
- In-memory sessions for MVP simplicity

---

## 📚 Notebooks

The `backend/notebooks/` directory contains exploration and prototyping work:
- Prompt engineering techniques
- Multi-LLM comparisons (GPT-4, Claude, Gemini)
- Vector RAG implementation
- LangChain agents and chains

---

## 🚢 Deployment

**Backend**: Railway → https://interview-ai-production-7e11.up.railway.app/  
**Frontend**: Vercel → https://interview-app-frontend-mu.vercel.app/

---

## 📄 License

MIT License

---

## 👤 Author

**Julian López**  
GitHub: [@jclopezsDS](https://github.com/jclopezsDS)