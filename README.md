# Hands-Off Operating Room

Hands-Off OR is a touchless dashboard for a scrubbed-in clinician. A webcam tracks the hands, a microphone listens for a spoken surgical log, and a dark, high-contrast viewer keeps images, notes, and nearby facility information on one screen. Nothing in this repository is a medical device. It is a hackathon prototype for demonstration, not for clinical use.

The dashboard opens at [http://127.0.0.1:3000](http://127.0.0.1:3000). The bridge that connects the camera, microphone, and viewer listens on port 8000.

## What you can do

- **Wake the room with a pinch.** Hold a pinch for about one second. The cursor is a ring while the hand is open and a solid dot while pinched. After about ten seconds with no hand in view, the interface goes back to sleep. The operation timer does not.
- **Point, click, and move films with the hand.** A pinch on a control is a click. Pinch and swipe up or down on a film to change images. Two hands pinching together zoom. A double pinch zooms to the point under the cursor.
- **Add and mark images.** Drop PNG, JPEG, WebP, GIF, or BMP files onto the viewer, or use **Add**. Draw mode lets a pinch trace on the current film. Strokes stay with that image. Clear drawing removes them.
- **Keep a surgical log by voice.** Say **log** to start a phrase and **exit** to finish it. The **Record** button does the same thing as saying **log**. Entries are numbered from the first one saved. The newest stays at the top. A grey **×** on each entry turns red under the hand cursor, and a pinch deletes that entry.
- **Time the operation.** The clock in the header starts the first time the interface leaves sleep. It keeps running if the room goes back to sleep, until the app is closed or **Stop operation** is pinched. That button turns red while the hand cursor is over it.
- **Look up nearby facilities.** With location permission and a SerpAPI key, the sidebar lists nearby medical facilities and the services named in their public listings. A pinch can scroll that list.

Spoken instrument commands can also be forwarded to Gemini Live, which may call `log_instrument` (added, removed, or used) and `verify_tray_count`.

## How the pieces connect

```text
Webcam ──► vision.py ──► hand position, pinch, and a low-rate video frame
Mic    ──► voice.py  ──► "log" / "exit" phrases, plus 16 kHz audio
                              │
                              ▼
                    FastAPI bridge  (ws://127.0.0.1:8000/ws)
                              │
          ┌───────────────────┼────────────────────┐
          ▼                   ▼                    ▼
   Next.js dashboard    Gemini Live session    Nearby-facility search
   http://127.0.0.1:3000
```

| Process | Role |
| --- | --- |
| `backend/server.py` | WebSocket bridge, tray tools, image picking, and the nearby-facility route |
| `backend/vision.py` | MediaPipe hand tracking. Optional: the viewer still runs if the camera is unavailable |
| `backend/voice.py` | Microphone listener. Optional: the viewer still runs if the mic is unavailable |
| `frontend` | The sterile dashboard |

`./start.sh` builds the frontend, then starts all four processes and opens the browser when both the page and `GET /healthz` respond.

## Requirements

- Node.js 20.9 or newer, with npm
- Python 3.11 or newer (`python3` on macOS and Linux, `py -3` on Windows)
- A webcam for hand tracking
- A microphone and an internet connection for speech recognition
- A Gemini API key if spoken audio should reach Gemini Live
- A [SerpAPI](https://serpapi.com/) key if nearby facilities should load

On a Mac, the vision and voice services prefer the built-in camera and microphone so a paired phone does not steal the input.

## Quick start

From the repository root:

```bash
./start.sh
```

On Windows, from the same folder:

```bat
npm start
```

The first run creates `backend/.venv`, installs Python and frontend dependencies, and builds the production dashboard. Later runs skip work that is already done. Leave the terminal open. Stopping it stops the bridge, camera, microphone, and dashboard together.

If port 3000 or 8000 is already taken, the start fails with `EADDRINUSE`. Quit the other process using that port, then run `./start.sh` again.

## Configuration

Create `backend/.env`. That file is gitignored.

```bash
# One of these lets the bridge open a Gemini Live session.
GEMINI_API_KEY=your-key
# GOOGLE_API_KEY=your-key
# GOOGLE_GENAI_API_KEY=your-key

# Optional. Defaults to gemini-3.8-live.
# GEMINI_LIVE_MODEL=gemini-3.8-live

# Required only for Nearby Medical Facilities.
SERPAPI_API_KEY=your-serpapi-key
```

The dashboard reads `NEXT_PUBLIC_BRIDGE_URL` when it is built. The default is `ws://localhost:8000/ws`. Change it only if the bridge is not on that address, and rebuild the frontend afterward.

| Variable | Default | Used by |
| --- | --- | --- |
| `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `GOOGLE_GENAI_API_KEY` | unset | Gemini Live |
| `GEMINI_LIVE_MODEL` | `gemini-3.8-live` | Gemini Live |
| `SERPAPI_API_KEY` | unset | `GET /api/nearby-medical` |
| `BRIDGE_HOST` | `127.0.0.1` | Bridge, when `server.py` is started directly |
| `BRIDGE_PORT` | `8000` | Bridge, when `server.py` is started directly |
| `BRIDGE_LOG_LEVEL` | `INFO` | Bridge |
| `NEXT_PUBLIC_BRIDGE_URL` | `ws://localhost:8000/ws` | Dashboard |

## Using the dashboard

1. Allow the camera and microphone when the operating system asks.
2. Hold a pinch until the header changes from **Asleep** to **Awake**. The operation clock starts on that first wake.
3. Move the open hand to point. Pinch a thumbnail, **Add**, **Record**, **Draw mode**, **Clear drawing**, a log **×**, or **Stop operation** to activate it. Controls turn to their highlight color while the hand cursor is over them.
4. Say **log**, speak the note, then say **exit**. A bare **log** waits for the next phrase. **Record** arms that same window.
5. To finish the timed operation, pinch **Stop operation**. The clock stays on the elapsed time until the app is closed.

Images you add stay in the browser tab for the session. They are not uploaded.

## Develop without the production build

Run the bridge from `backend` after the virtualenv exists:

```bash
cd backend
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m uvicorn server:app --port 8000 --reload
```

In other terminals, still inside `backend` with the virtualenv active:

```bash
python vision.py
python voice.py
```

Then start the dashboard:

```bash
cd frontend
npm install
npm run dev
```

`npm run dev` also tries to launch the wake-word listener when one is not already running. Open [http://127.0.0.1:3000](http://127.0.0.1:3000).

Useful bridge routes:

| Route | Purpose |
| --- | --- |
| `GET /healthz` | Liveness check used by `./start.sh` |
| `GET /api/state` | Latest bridge snapshot |
| `GET /api/gemini/tools` | Tool declarations sent to Gemini |
| `GET /api/nearby-medical` | Facility search |
| `WS /ws` | Telemetry, speech-log frames, audio, and video |

Vision and voice connect as input clients. The dashboard connects as a viewer.

## Tests

Frontend tests use Node's built-in test runner:

```bash
cd frontend
node --experimental-strip-types --test tests/*.test.ts
```

Backend tests:

```bash
cd backend
python -m unittest discover -s tests -v
```

## Repository layout

```text
start.sh / start.js          one-command setup and launch
frontend/                    Next.js dashboard
  app/page.tsx               viewer, log, timer, and hand gestures
  app/speech-log-state.ts    surgical-log state
  app/operation-timer.ts     operation clock
  components/                image viewer, drawing, and shape snap
backend/
  server.py                  FastAPI bridge
  vision.py                  camera and MediaPipe hands
  voice.py                   microphone and wake word
  speech_log.py              log / exit session
  gemini_live.py             Gemini Live audio session
  tools.py                   log_instrument and verify_tray_count
  shared_schema.json         messages shared by the bridge and the dashboard
```

## Privacy and safety

Camera frames, microphone audio, and the facility search leave the machine only for the services you configure: Google speech recognition for the wake word, Gemini Live when an API key is set, and SerpAPI when nearby facilities are requested. The browser asks before sharing location. Treat API keys as secrets and keep them in `backend/.env`.

Do not rely on this prototype for a count, a diagnosis, or a clinical decision. Hand tracking and speech recognition miss words and gestures, especially in a loud room.
