# IntervueAI Capstone

IntervueAI is an AI-powered mock interview platform that allows candidates to upload their resume and start an automated voice interview based on a given job description (JD). The system combines a FastAPI backend, a LiveKit-powered voice assistant agent, and a frontend UI built using Next.js and `pnpm`.

## ✅ Features Implemented

### 🎤 Voice-based Interview Agent
**WebRTC framework (LiveKit SDK):** Built a real-time voice-based interview system using LiveKit SDK with Python backend and React frontend.

#### Tech Stack:
- **Frontend:** React.js + LiveKit SDK
- **Backend:** FastAPI + Livekit API (Python)
- **LLM:** Gemini 2.0 Flash (Google)
- **Voice:** AWS Text-to-Speech (TTS) and Speech-to-Text (STT)


### 🧠 Candidate Screening Logic
**Resume Parser & JD Matching Agent:** Automatically accepts or rejects candidates before starting the interview by evaluating JD-candidate fit using vector similarity and heuristics.

### 💻 Coding Interview Support
**Integrated Coding Editor:** For coding, DSA, or algorithm questions, the agent asks the candidate to open an embedded code editor to write and explain their solution.

### 🔇 Silence Handling
**Passive Listening:** Agent intelligently stays quiet if the candidate is thinking or pausing, ensuring a natural human-like experience.

**Deadlock Resolution:** If both agent and candidate are silent beyond a threshold, the agent breaks the silence with conversational nudges (e.g., “Take your time,” or “Do you need a hint?”).

### 🔍 Probing & Evaluation Logic
**Contextual Probing:** Follow-up questions are dynamically decided based on the response depth, evaluation weight of the question, and available time.

**Stuck Detection & Hints:** Agent can detect if a candidate is stuck and optionally provide contextual hints to assist progress (configurable).

### 📝 Feedback Report Generation
At the end of each interview, the system generates a structured, rubric-based feedback report for the candidate — useful for self-evaluation and interview preparation.

### 📊 Pretraining Strategy
[Refer to the detailed README here](pre_training/README.md)

**Pretrained Foundation Model:** Llama3 1B.

**Fine-tuning:** Applied light fine-tuning on a domain-specific dataset of interview questions and candidate responses to align the model better with real-world interview patterns.

### ⚠️ Current Challenges & Areas for Improvement

**Prompt Engineering for Custom Models:** Initial tests with local LLMs via Ollama (Phi, Mistral, etc.) didn’t yield great results. Need improved prompt chaining or agent design.

**Dependency on AWS STT/TTS:** Chosen due to available credits and faster development, but may explore alternatives (e.g., Whisper, Coqui TTS) for cost scalability or latency.

**Latency and Agent Responsiveness:** Still tuning for natural response pacing and reducing perceived lag in turn-taking.

---

## 🧱 Project Structure

```
IntervueAI_Capstone/
├── README.md
├── requirements.txt
├── .env.local
├── main.py                     # Main entrypoint
├── configs/
│   ├── livekit.yaml           # Needed for production setup
│   └── config.yaml            # Model, path, port configs
│
├── agents/
│   ├── __init__.py
│   ├── interview_agent.py      # Manages session and invokes coordinator agent
│   ├── feedback.py             # Feedback Agent to generate
│   ├── resume_pdf_parser.py    # Resume Parser Agent
│   ├── jd_resume_matcher.py    # JD <-> Candidate's Resume Matcher Agent
│   ├── interview_planner.py    # Interview Planner Agent
│   └── coordinator.py          # Coordinator Agent that Orchestrates and conducts interview

├── prompts/                   # Prompt templates for each agent
├── ui/
│   ├── web_client/            # React/Next.js web client for LiveKit frontend
│       └── components/        # Chat bubbles, code editor, mic controls
└── utils/
│   ├── __init__.py
│   ├── session.py             # interview session utility
│   ├── memory.py              # interview conversation memory
│   └── llm.py                 # LLM calls utility
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

### 🗣️ 4. LiveKit Server – Local Development Setup

This project uses [LiveKit](https://livekit.io/) to enable real-time voice communication for interview agents. You can ignore this step and
use livekit cloud (FREEMIUM) by Signig up here [Livekit Cloud](https://livekit.io) 

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
GOOGLE_API_KEYS=[<list_of_google_api_keys_selected_based_on_rotation]
GOOGLE_API_KEY=<google_api_key_for_running_llm>
```

## 📦 Recommended Tools

- pyenv — for managing Python versions
- pnpm — faster, disk-efficient alternative to npm
- LiveKit — real-time voice and video infrastructure
