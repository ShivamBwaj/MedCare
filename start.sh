#!/bin/bash

echo "Starting TrialChain+ColdCare..."

# Start backend
echo "Starting FastAPI backend..."
cd backend
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "Starting React frontend..."
cd ../frontend
npm start &
FRONTEND_PID=$!

echo "Application started!"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"

# Wait for user to stop
echo "Press Ctrl+C to stop all services"
wait
