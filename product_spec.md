# Hands-Off OR: Product Specification

## 1. Project Pitch & Overview
* The Problem: Surgeons in under-resourced clinics often lack circulating nurses to help navigate medical records, DICOM viewers, or surgical logs while scrubbed in. 
* The Solution: A multimodal, touchless surgical assistant leveraging local edge AI for real-time hand tracking (30 FPS) and the Gemini Live API for voice/visual reasoning and function calling.

## 2. System Architecture
[Webcam & Mic] 
   │
   ├─> (30 FPS Video) ──> [OpenCV + MediaPipe Edge Node] ──> Extracts (x, y, pinch) ──┐
   │                                                                                  │
   └─> (1 FPS Video + Audio) ──> [Audio/Video Buffer] ──> [Gemini Live API Node] ─────┤
                                                          (Multimodal Reasoning)      │
                                                                                      ▼
[React/Next.js Frontend] <══(Unified Telemetry & Function Call Stream)══ [FastAPI WebSocket Bridge]

## 3. React Dashboard UI Mockup (The Sterile Interface)
* Dark Mode Design: High contrast to reduce eye strain and make scans pop.
* Main Viewport (Left/Center): A large interactive DICOM/X-ray viewer area that responds to the pinch-to-pan and pinch-to-zoom simulated mouse events.
* Right Sidebar: A live-updating 'Surgical Log' and tool count driven by the Gemini API function calls.
* Floating Cursor: A custom div fixed to the screen that tracks the {x,y} coordinates. It should look like a glowing hollow ring, and instantly change to a solid dot when {pinch_active} is true.
* Voice & Audio Pipeline: A parallel Python script that listens to the microphone. Upon hearing the wake word "Assistant," it captures the audio, streams it to the Gemini Live API via the WebSocket bridge, and handles the log_instrument and verify_tray_count function calls to update the React UI.
