# IntervueAI Capstone

IntervueAI is an AI-powered mock interview platform that allows candidates to upload their resume and start an automated voice interview based on a given job description (JD). The system combines a FastAPI backend, a LiveKit-powered voice assistant agent, and a frontend UI built using Next.js and `pnpm`.

---

## 🧱 Project Structure

```
IntervueAI_Capstone/
├── README.md
├── requirements.txt
├── run.py                     # Main entrypoint
├── configs/
│   └── config.yaml            # Model, path, port configs
├── agents/
│   ├── __init__.py
│   ├── macro_planner.py       # Subagent 1 - JD + Resume to Questionnaire
│   ├── interview_loop.py      # Subagent 2 - Interview FSM Controller
│   ├── scorer_summary.py      # Subagent 3 - Evaluation and summary
│   └── session_manager.py     # Subagent 4 - Orchestrates session lifecycle
├── fsm/
│   ├── __init__.py
│   └── interview_states.py    # FSM states for the main interview loop
├── services/
│   ├── __init__.py
│   ├── audio_service.py       # STT and TTS services (Whisper, Coqui, etc.)
│   ├── llm_interface.py       # Wrapper around LLaMA model (vLLM or transformers)
│   ├── vector_store.py        # Resume/JD embeddings + context memory
│   └── livekit_client.py      # Interface to LiveKit video/audio session control
├── data/
│   ├── prompts/               # Prompt templates for each agent
│   └── examples/              # Sample JDs, resumes, transcripts
├── ui/
│   ├── web_client/            # React/Next.js web client for LiveKit frontend
│       └── components/            # Chat bubbles, code editor, mic controls
└── utils/
    ├── __init__.py
    ├── logger.py              # Unified logger
    ├── schema.py              # Dataclasses or Pydantic models for Q&A, Plan, Report
    └── helpers.py             # Misc utilities
```


---

## 🚀 Getting Started

### 🐍 1. Backend Setup (FastAPI)

#### 📦 Install dependencies (recommended: virtualenv or pyenv)

```bash
pyenv virtualenv interview-agent-env
source interview-agent-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### ⚙️ Run FastAPI server
```
uvicorn main:app --reload --port 8000
```
FastAPI will now be running at:

```📍 http://localhost:8000```

### 🤖 2. Agent Setup (LiveKit Voice Assistant)
This agent listens in a LiveKit room and conducts the mock interview using TTS and STT.
Ensure AWS credentials are set in your .env.local or environment variables.

```
# Start the agent subprocess manually
python agents/interview_agent.py dev
```

This will load the entrypoint() function, connect to a LiveKit room, and start voice interactions when a participant joins.

### 💻 3. Frontend Setup (Next.js using pnpm)
#### 📦 Install dependencies
```
cd ui/web_client
pnpm install
```

#### ▶️ Start the dev server

```
pnpm dev
```
Frontend will run at:
```📍 http://localhost:3000```


## 🔗 End-to-End Flow

1. User visits job description page → clicks Apply
2. User uploads resume → backend returns roomToken and serverUrl
3. User is redirected to the voice interview page using those credentials
4. LiveKit agent joins the room → starts voice interview automatically

## 🛠️ Environment Variables

Create a .env.local file for the backend and agent:

```
# .env.local
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=ap-south-1

LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
LIVEKIT_URL=https://your-livekit-server.com
```

## 📦 Recommended Tools

- pyenv — for managing Python versions
- pnpm — faster, disk-efficient alternative to npm
- LiveKit — real-time voice and video infrastructure
