#!/bin/bash
echo "Starting Surgical Assistant..."

# Trap Ctrl+C and kill all child background processes automatically
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Start backend services in the background
cd backend
uvicorn server:app --port 8000 &
python3 vision.py &
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi
export HANDFREEOR_VOICE_STARTED=1
"$PY" voice.py &

# Start frontend in the foreground
cd ../frontend
open http://localhost:3000
npm run dev