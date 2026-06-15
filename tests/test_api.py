from __future__ import annotations

import base64
import io
import sys
import unittest
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ed_platform.app import app


FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "minimal_case"
PASS_FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "minimal_case_pass"


def _encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    def _folder_payload(self, fixture_root: Path, label: str) -> dict[str, object]:
        files = []
        for path in sorted(fixture_root.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(fixture_root).as_posix(),
                        "content_base64": _encode_file(path),
                    }
                )
        return {
            "label": label,
            "bc_y": "PBC",
            "max_basis_states": 1000,
            "files": files,
        }

    def _zip_payload(self, fixture_root: Path, label: str) -> dict[str, object]:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(fixture_root.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(fixture_root).as_posix())
        return {
            "label": label,
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

    def test_validate_folder_payload_returns_failures(self) -> None:
        response = self.client.post("/api/validate", json=self._folder_payload(FIXTURE_ROOT, "fixture-folder"))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["overall_status"], "fail")
        self.assertEqual(payload["case_count"], 1)
        self.assertEqual(len(payload["cases"]), 1)
        case = payload["cases"][0]
        self.assertEqual(case["case"], "Lx2Ly1_d0_u4_tp0_N2")
        self.assertEqual(case["status"], "fail")
        self.assertGreaterEqual(len(case["failures"]), 1)
        self.assertEqual(case["checks"][0]["status"], "skipped")
        self.assertEqual(case["checks"][1]["status"], "fail")

    def test_validate_pass_fixture(self) -> None:
        response = self.client.post("/api/validate", json=self._folder_payload(PASS_FIXTURE_ROOT, "fixture-pass"))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["overall_status"], "pass")
        self.assertEqual(payload["summary"]["passed_cases"], 1)
        case = payload["cases"][0]
        self.assertEqual(case["status"], "pass")
        self.assertEqual(case["checks"][0]["status"], "skipped")
        self.assertEqual(case["checks"][1]["status"], "pass")

    def test_validate_zip_payload(self) -> None:
        response = self.client.post("/api/validate", json=self._zip_payload(PASS_FIXTURE_ROOT, "fixture-zip"))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["case_count"], 1)
        self.assertEqual(payload["overall_status"], "pass")


if __name__ == "__main__":
    unittest.main()
