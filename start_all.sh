
#!/bin/bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate interview-agent-env

cd /home/ubuntu/IntervueAI_Capstone

# Start LiveKit server in background
livekit-server --config configs/livekit.yaml > livekit.log 2>&1 &

# Start backend in background
uvicorn main:app --host 0.0.0.0 --port 8000 > fastapi.log 2>&1 &

# Start frontend in background
cd ui/web_client
pnpm dev > frontend.log 2>&1 &

# Back to root folder
cd ../..

# Keep agent in foreground
exec python -m agents.interview_agent start