import asyncio
import json
import unittest

from server import Client, bridge, handle_text
from tools import clear_tray, interpret_log_command, live_function_declarations


class TestInterpretLogCommand(unittest.TestCase):
    def test_add_and_quantity(self):
        self.assertEqual(
            interpret_log_command("add sponge and two clamps"),
            [
                ("log_instrument", {"instrument": "sponge", "action": "added", "quantity": 1}),
                ("log_instrument", {"instrument": "clamps", "action": "added", "quantity": 2}),
            ],
        )

    def test_remove_and_verify(self):
        self.assertEqual(
            interpret_log_command("removed 3 forceps"),
            [("log_instrument", {"instrument": "forceps", "action": "removed", "quantity": 3})],
        )
        self.assertEqual(
            interpret_log_command("verify 12"),
            [("verify_tray_count", {"expected_count": 12})],
        )
        self.assertEqual(
            interpret_log_command("verify 4 sponges"),
            [("verify_tray_count", {"expected_count": 4, "instrument": "sponges"})],
        )

    def test_live_declarations_are_callable(self):
        declarations = live_function_declarations()
        self.assertEqual(
            [item["name"] for item in declarations],
            ["log_instrument", "verify_tray_count"],
        )
        quantity = declarations[0]["parameters"]["properties"]["quantity"]
        self.assertNotIn("minimum", quantity)
        self.assertEqual(declarations[0]["behavior"], "BLOCKING")


class TestSpokenLogReachesViewer(unittest.TestCase):
    def test_final_transcript_is_stored_and_broadcast(self):
        async def run_test():
            clear_tray()
            viewer_frames: list[dict] = []

            async def viewer_send(data: str) -> None:
                viewer_frames.append(json.loads(data))

            async def ignore(_: str) -> None:
                return None

            class Socket:
                def __init__(self, send):
                    self._send = send

                async def send_text(self, data: str) -> None:
                    await self._send(data)

            voice = Client("voice-log", "input", Socket(ignore), 0)
            viewer = Client("viewer-log", "viewer", Socket(viewer_send), 0)
            await bridge.add(voice)
            await bridge.add(viewer)
            try:
                await handle_text(
                    voice,
                    json.dumps({"type": "speech_log", "phase": "final", "text": "add sponge"}),
                )
            finally:
                await bridge.remove("voice-log")
                await bridge.remove("viewer-log")

            calls = [frame for frame in viewer_frames if frame.get("type") == "function_call"]
            results = [frame for frame in viewer_frames if frame.get("type") == "function_result"]
            self.assertEqual(len(calls), 1, viewer_frames)
            self.assertEqual(calls[0]["name"], "log_instrument")
            self.assertEqual(calls[0]["args"]["instrument"], "sponge")
            self.assertEqual(calls[0]["args"]["action"], "added")
            self.assertEqual(len(results), 1, viewer_frames)
            self.assertTrue(results[0]["ok"])
            self.assertEqual(results[0]["result"]["on_tray"], 1)
            self.assertEqual(results[0]["call_id"], calls[0]["call_id"])

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
