from __future__ import annotations

import base64
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ed_platform.app import app
from ed_platform import service


FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "minimal_case"


def _encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        service.STORAGE_ROOT = Path(self.temp_dir.name) / "runs"
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.temp_dir.cleanup()

    def _folder_payload(self) -> dict[str, object]:
        files = []
        for path in sorted(FIXTURE_ROOT.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(FIXTURE_ROOT).as_posix(),
                        "content_base64": _encode_file(path),
                    }
                )
        return {
            "label": "fixture-folder",
            "bc_y": "PBC",
            "max_basis_states": 1000,
            "files": files,
        }

    def _zip_payload(self) -> dict[str, object]:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(FIXTURE_ROOT.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(FIXTURE_ROOT).as_posix())
        return {
            "label": "fixture-zip",
            "bc_y": "PBC",
            "max_basis_states": 1000,
            "files": [
                {
                    "path": "minimal-case.zip",
                    "content_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                }
            ],
        }

    def test_healthcheck(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_validate_folder_payload(self) -> None:
        response = self.client.post("/api/validate", json=self._folder_payload())
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["case_count"], 1)
        self.assertEqual(len(payload["cases"]), 1)
        case = payload["cases"][0]
        self.assertEqual(case["case"], "Lx2Ly1_d0_u4_tp0_N2")
        self.assertEqual(case["green_status"], "compared")
        self.assertIsNotNone(case["qmc_energy"])
        self.assertIsNotNone(case["green_relative_frobenius_up"])
        self.assertIn("report_json", payload["downloads"])

    def test_validate_zip_payload(self) -> None:
        response = self.client.post("/api/validate", json=self._zip_payload())
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        run_id = payload["run_id"]
        detail_response = self.client.get(f"/api/runs/{run_id}")
        self.assertEqual(detail_response.status_code, 200)
        self.assertEqual(detail_response.json()["run_id"], run_id)
        artifact_response = self.client.get(f"/api/runs/{run_id}/artifacts/report_json")
        self.assertEqual(artifact_response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
