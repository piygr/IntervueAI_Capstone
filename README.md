project_root/
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

    