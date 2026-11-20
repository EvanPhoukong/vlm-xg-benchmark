"""Regression tests for launching the checker with zero third-party packages."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PreflightTests(unittest.TestCase):
    def run_without_site_packages(self, execute=False):
        script = Path(__file__).with_name("train.py")
        with tempfile.TemporaryDirectory() as directory:
            status = Path(directory)/"status.json"
            command = [sys.executable,"-S",str(script),"--status-file",str(status)]
            if execute:
                command += ["--execute","--smoke"]
            process = subprocess.run(command,capture_output=True,text=True,timeout=45)
            self.assertNotIn("Traceback",process.stderr)
            self.assertTrue(status.is_file(),process.stderr)
            result = json.loads(status.read_text())
            self.assertEqual(result["status"],"blocked_missing_training_runtime")
            self.assertIn("numpy",result["missing_packages"])
            self.assertIn("torch",result["missing_packages"])
            self.assertEqual(result["paid_api_calls"],0)
            self.assertEqual(result["python_executable"],sys.executable)
            return process

    def test_checker_reports_missing_numpy_instead_of_crashing(self):
        self.assertEqual(self.run_without_site_packages().returncode,0)

    def test_execute_stops_cleanly_before_importing_training_helpers(self):
        self.assertEqual(self.run_without_site_packages(execute=True).returncode,2)


if __name__ == "__main__":
    unittest.main()
