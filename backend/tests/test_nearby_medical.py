import asyncio
import base64
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException

from server import (
    _NEARBY_CACHE,
    _load_dotenv_file,
    _local_ll,
    _service_notes_from_text,
    _top_local_results,
    nearby_medical,
)


class TestNearbyMedical(unittest.TestCase):
    def setUp(self):
        _NEARBY_CACHE.clear()

    def test_ll_packs_latitude_longitude_and_zoom(self):
        self.assertEqual(_local_ll(40.712776, -74.005974), "@40.712776,-74.005974,14z")

    def test_returns_at_most_five_facilities_with_distance(self):
        payload = {
            "local_results": [
                {
                    "title": f"Place {index}",
                    "address": f"{index} Main St",
                    "gps_coordinates": {"latitude": 40.0 + index * 0.01, "longitude": -74.0},
                }
                for index in range(8)
            ]
        }
        results = _top_local_results(payload, 40.0, -74.0)
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0]["name"], "Place 0")
        self.assertEqual(results[0]["address"], "0 Main St")
        self.assertEqual(results[0]["distance"], "0 ft")
        self.assertEqual(results[1]["distance"], "0.7 mi")
        self.assertEqual(results[4]["name"], "Place 4")

    def test_keeps_a_distance_supplied_by_the_search(self):
        results = _top_local_results(
            {
                "local_results": [
                    {"title": "City Hospital", "address": "1 Oak", "distance": "0.4 mi"}
                ]
            },
            40.0,
            -74.0,
        )
        self.assertEqual(results[0]["distance"], "0.4 mi")

    def test_service_notes_use_only_published_phrases(self):
        notes = _service_notes_from_text(
            "Dell Seton Medical Center at The University of Texas",
            "Hospital",
            "Open 24 hours",
            [
                "Dell Seton is a Level I Trauma center with stroke care and 24/7 emergency care."
            ],
        )
        self.assertEqual(notes, ["Level I trauma center", "Stroke care"])

    def test_service_notes_skip_a_directory_mention_of_imaging(self):
        notes = _service_notes_from_text(
            "Dell Seton Medical Center at The University of Texas - Emergency",
            "Emergency room",
            "Open 24 hours",
            ["With a Level I Trauma Center, the emergency department delivers 24/7 emergency care."],
        )
        self.assertEqual(notes, ["Level I trauma center", "24/7 emergency care"])

    def test_service_notes_keep_telemedicine_for_urgent_care(self):
        notes = _service_notes_from_text(
            "Austin Online Urgent Care",
            "Urgent care center",
            "Open 24 hours",
            ["Telemedicine lets you see a provider for urgent care."],
        )
        self.assertEqual(notes, ["Telemedicine", "Urgent care"])

    def test_service_notes_stay_empty_when_nothing_is_named(self):
        notes = _service_notes_from_text(
            "Saint Mary Medical Center",
            "Medical clinic",
            "",
            ["Customers can explore more in store and get healthcare solutions."],
        )
        self.assertEqual(notes, [])

    def test_search_uses_google_local_and_hides_the_api_key(self):
        captured: dict = {}

        def fake_search(params):
            captured.update(params)
            return {
                "local_results": [
                    {"title": "Riverside Clinic", "address": "9 River Rd", "distance": "1.2 mi"},
                    {"name": "", "address": "skip me"},
                ]
            }

        with (
            patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"}),
            patch("server._google_local_search", fake_search),
            patch("server._attach_service_notes", side_effect=lambda facilities, _key: [
                {
                    "name": item["name"],
                    "distance": item["distance"],
                    "address": item["address"],
                    "services": ["Primary care", "Preventive care"],
                }
                for item in facilities
            ]),
        ):
            body = asyncio.run(nearby_medical(40.0, -74.0))

        self.assertEqual(captured["engine"], "google_local")
        self.assertEqual(captured["q"], "hospital clinic")
        self.assertEqual(captured["ll"], "@40.000000,-74.000000,14z")
        self.assertEqual(captured["gl"], "us")
        self.assertEqual(captured["api_key"], "test-key")
        decoded = base64.b64decode(captured["uule"][2:]).decode()
        self.assertIn("latitude_e7:400000000", decoded)
        self.assertIn("longitude_e7:-740000000", decoded)
        self.assertEqual(
            body["local_results"],
            [
                {
                    "name": "Riverside Clinic",
                    "distance": "1.2 mi",
                    "address": "9 River Rd",
                    "services": ["Primary care", "Preventive care"],
                }
            ],
        )
        self.assertNotIn("api_key", body)

    def test_dotenv_fills_only_missing_keys(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('SERPAPI_API_KEY="from-file"\nALREADY=new\n# ignored\n')
            with patch.dict(os.environ, {"ALREADY": "kept"}, clear=True):
                _load_dotenv_file(path)
                self.assertEqual(os.environ["SERPAPI_API_KEY"], "from-file")
                self.assertEqual(os.environ["ALREADY"], "kept")

    def test_missing_key_is_rejected(self):
        env = os.environ.copy()
        env.pop("SERPAPI_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(nearby_medical(40.0, -74.0))
        self.assertEqual(raised.exception.status_code, 503)

    def test_search_error_is_rejected_without_echoing_the_key(self):
        def fake_search(_params):
            return {"error": "Invalid API key"}

        with (
            patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"}),
            patch("server._google_local_search", fake_search),
        ):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(nearby_medical(40.0, -74.0))
        self.assertEqual(raised.exception.status_code, 502)
        self.assertNotIn("test-key", str(raised.exception.detail))


if __name__ == "__main__":
    unittest.main()
