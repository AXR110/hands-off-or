import json
import unittest
import asyncio
from server import handle_text, bridge, Client


class TestSpeechLogRelay(unittest.TestCase):
    def test_speech_log_relay_from_input(self):
        async def run_test():
            voice_frames = []
            viewer_frames = []

            async def voice_send(data):
                voice_frames.append(json.loads(data))

            async def viewer_send(data):
                viewer_frames.append(json.loads(data))

            class Socket:
                def __init__(self, send):
                    self._send = send

                async def send_text(self, data):
                    await self._send(data)

            voice_client = Client("voice-1", "input", Socket(voice_send), 0)
            viewer_client = Client("viewer-1", "viewer", Socket(viewer_send), 0)

            await bridge.add(voice_client)
            await bridge.add(viewer_client)
            try:
                # Case 1: valid final from input
                await handle_text(voice_client, json.dumps({
                    "type": "speech_log",
                    "phase": "final",
                    "text": "add sponge"
                }))
                # Assert broadcast to viewer only
                speech_logs = [f for f in viewer_frames if f.get("type") == "speech_log"]
                self.assertEqual(len(speech_logs), 1, f"viewer_frames: {viewer_frames}")
                log = speech_logs[0]
                self.assertEqual(log["phase"], "final")
                self.assertEqual(log["text"], "add sponge")
                self.assertEqual(log["source"], "voice-1")
                self.assertIsInstance(log.get("server_ts"), int)
                # sender did not receive copy
                voice_speech = [f for f in voice_frames if f.get("type") == "speech_log"]
                self.assertEqual(len(voice_speech), 0, f"voice_frames: {voice_frames}")

                # Case 2: viewer sends -> forbidden_role
                viewer_frames.clear()
                voice_frames.clear()
                await handle_text(viewer_client, json.dumps({
                    "type": "speech_log",
                    "phase": "final",
                    "text": "add sponge"
                }))
                self.assertEqual(len([f for f in viewer_frames if f.get("type") == "speech_log"]), 0)
                self.assertEqual(len([f for f in voice_frames if f.get("type") == "speech_log"]), 0)
                errors = [f for f in viewer_frames if f.get("type") == "error" and f.get("code") == "forbidden_role"]
                self.assertTrue(len(errors) >= 1, f"viewer_frames: {viewer_frames}")

                # Case 3: blank text from input -> invalid_payload
                viewer_frames.clear()
                voice_frames.clear()
                await handle_text(voice_client, json.dumps({
                    "type": "speech_log",
                    "phase": "final",
                    "text": "   "
                }))
                self.assertEqual(len([f for f in viewer_frames if f.get("type") == "speech_log"]), 0)
                self.assertEqual(len([f for f in voice_frames if f.get("type") == "speech_log"]), 0)
                errors = [f for f in voice_frames if f.get("type") == "error" and f.get("code") == "invalid_payload"]
                self.assertTrue(len(errors) >= 1, f"voice_frames: {voice_frames}")
            finally:
                await bridge.remove("voice-1")
                await bridge.remove("viewer-1")

        asyncio.run(run_test())

    def test_voice_status_reaches_viewers(self):
        async def run_test():
            viewer_frames = []

            async def viewer_send(data):
                viewer_frames.append(json.loads(data))

            async def ignore(_data):
                return None

            class Socket:
                def __init__(self, send):
                    self._send = send

                async def send_text(self, data):
                    await self._send(data)

            voice = Client("voice-status", "input", Socket(ignore), 0)
            viewer = Client("viewer-status", "viewer", Socket(viewer_send), 0)
            await bridge.add(voice)
            await bridge.add(viewer)
            try:
                await handle_text(
                    voice,
                    json.dumps({"type": "voice_status", "active": True, "level": 0.65}),
                )
            finally:
                await bridge.remove("voice-status")
                await bridge.remove("viewer-status")
            status = [frame for frame in viewer_frames if frame.get("type") == "voice_status"]
            self.assertEqual(len(status), 1, viewer_frames)
            self.assertTrue(status[0]["active"])
            self.assertEqual(status[0]["level"], 0.65)
            self.assertEqual(status[0]["source"], "voice-status")

        asyncio.run(run_test())

    def test_record_controls_relay_to_input_clients(self):
        async def run_test():
            input_frames = []
            viewer_frames = []

            async def input_send(data):
                input_frames.append(json.loads(data))

            async def viewer_send(data):
                viewer_frames.append(json.loads(data))

            class Socket:
                def __init__(self, send):
                    self._send = send

                async def send_text(self, data):
                    await self._send(data)

            input_client = Client("record-input", "input", Socket(input_send), 0)
            viewer_client = Client("record-viewer", "viewer", Socket(viewer_send), 0)
            await bridge.add(input_client)
            await bridge.add(viewer_client)
            try:
                for command in ("start_recording", "stop_recording"):
                    await handle_text(viewer_client, json.dumps({"type": command}))
                self.assertEqual(
                    [frame["type"] for frame in input_frames],
                    ["start_recording", "stop_recording"],
                )
                self.assertEqual(viewer_frames, [])
            finally:
                await bridge.remove("record-input")
                await bridge.remove("record-viewer")

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
