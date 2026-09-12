# PURIVU — AI-Powered Intelligent Assistant for Indian Standards & BIS Services

**PURIVU** is a lightweight, high-precision compliance and intelligence assistant built for Indian Standards (IS), Bureau of Indian Standards (BIS) regulations, Quality Control Orders (QCOs), and certification services. Designed with a warm, accessible light-theme identity, PURIVU simplifies standards exploration for manufacturers, consumers, and regulatory officials.

---

## 🌟 Key Features

1. **ASK Mode (Text RAG Query)**:
   - Grounded RAG search over official BIS regulatory documents and guidelines.
   - Vector L2 distance matching with HuggingFace `all-MiniLM-L6-v2` embeddings and FAISS index.
   - Multi-query expansion and citation of page numbers, IS numbers, clauses, and scheme numbers.

2. **SHOW Mode (PURIVU Vision Engine)**:
   - Identifies physical products from uploaded images using Gemini Vision.
   - Queries the local BIS vector database to retrieve mandatory certification standards, QCOs, and compliance requirements.
   - Allows inline product correction.

3. **VAANI Mode (Voice & Multilingual Assistant)**:
   - Web Speech API integration supporting real-time voice input in **English**, **Tamil (தமிழ்)**, and **Hindi (हिन्दी)**.
   - Strict multilingual answer generation preserving technical BIS identifiers (e.g. `BIS`, `IS 15820`, `QCO`, `CoC`, `Scheme-I`, `Scheme-IV`) in original script.
   - Response-language aware Text-to-Speech (`ta-IN`, `hi-IN`, `en-IN`).

---

## 🏗️ Architecture

```
React Vite Frontend (Port 5173)
        │
        ▼  REST API (JSON / FormData)
FastAPI Backend (Port 8000)
        │
        ▼
PURIVU RAG Engine ──► FAISS Vector Index (data/bis/ PDFs)
        │
        ▼
Google Gemini API (Grounded Answer & Multilingual Generation)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 1. Environment Setup

Copy `.env.example` to `.env` and configure your Gemini API Key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

### 2. Backend Setup & Run

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

Run the FastAPI backend server:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

- API Base URL: `http://127.0.0.1:8000`
- Swagger Documentation: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/api/health`

### 3. Frontend Setup & Run

In a separate terminal, navigate to the `frontend` directory:

```bash
cd frontend
npm install
npm run dev
```

Open your browser at `http://127.0.0.1:5173`.

---

## 📄 License & Attribution

Developed for **Smart India Hackathon (SIH)** — Indian Standards & BIS Intelligence Assistant.
