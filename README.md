# IntervueAI Capstone

IntervueAI is an AI-powered mock interview platform that allows candidates to upload their resume and start an automated voice interview based on a given job description (JD). The system combines a FastAPI backend, a LiveKit-powered voice assistant agent, and a frontend UI built using Next.js and `pnpm`.

---

## 🧱 Project Structure

```
IntervueAI_Capstone/
├── README.md
├── requirements.txt
├── .env.local
├── main.py                     # Main entrypoint
├── configs/
│   └── config.yaml            # Model, path, port configs
├── agents/
│   ├── __init__.py
    ├── interview_agent.py
│   ├── macro_planner.py       # tentative Subagent 1 - JD + Resume to Questionnaire
│   ├── interview_loop.py      # tentative Subagent 2 - Interview FSM Controller
│   ├── scorer_summary.py      # tentative Subagent 3 - Evaluation and summary
│   └── session_manager.py     # tentative Subagent 4 - Orchestrates session lifecycle
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
│
├── pre_training/              # Directory containing pre-training files 
│   └── README.md              # Read me info specific for pretrainig part
│   ├── models/            
│       └── llama3.py  



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
# Start the agent subprocess manually from the project_root directory
python -m agents.interview_agent dev
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

### 🗣️ 4. LiveKit Server – Local Development Setup (Optional)

This project uses [LiveKit](https://livekit.io/) to enable real-time voice communication for interview agents. You can ignore this step and
use livekit cloud by Signig up here [Livekit Cloud](https://livekit.io) 

#### 🔧 Install LiveKit Server Self-hosting (locally)

##### On Mac
```
brew install livekit
```

##### On other OS
Refer here - [Livekit Server Self-hosting](https://docs.livekit.io/home/self-hosting/local/)

#### ▶️ Run LiveKit Server in Dev Mode
To start the server locally:
```
livekit-server --dev
```
This will:
- Start the server on ws://localhost:7880
- Print API_KEY and API_SECRET in your terminal (devkey / secret)
- Serve the REST API at http://localhost:7880

## 🔗 End-to-End Flow

1. User visits job description page → clicks Apply
2. User uploads resume → backend returns roomToken and serverUrl
3. User is redirected to the voice interview page using those credentials
4. LiveKit agent joins the room → starts voice interview automatically

## 🛠️ Environment Variables

Create a .env.local file for the backend and agent:

```
# .env.local
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
AWS_ACCESS_KEY_ID=<aws_access_key>
AWS_SECRET_ACCESS_KEY=<aws_access_key_secret>
AWS_REGION=<aws_region>
GOOGLE_API_KEY=<google_api_key_for_running_llm>
```

## 📦 Recommended Tools

- pyenv — for managing Python versions
- pnpm — faster, disk-efficient alternative to npm
- LiveKit — real-time voice and video infrastructure
