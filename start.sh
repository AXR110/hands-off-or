#!/bin/bash
echo "Starting Surgical Assistant..."

# Trap Ctrl+C and kill all child background processes automatically
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Start backend services in the background
cd backend
uvicorn server:app --port 8000 &
python3 vision.py &
python3 voice.py &

# Start frontend in the foreground
cd ../frontend
npm run dev