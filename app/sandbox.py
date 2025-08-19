# sandbox.py
import tempfile
import subprocess
import os
from typing import Dict, Tuple

def run_code_in_sandbox(code_str: str, files: Dict[str, bytes]) -> Tuple[str, str]:
    """
    Runs `code_str` inside a temp dir with provided `files` written to disk.
    Returns (stdout, stderr) as strings.
    No internet – just local execution.
    """
    with tempfile.TemporaryDirectory() as td:
        # write attached files
        for name, content in (files or {}).items():
            # Ensure subdirs exist if name contains path separators
            full = os.path.join(td, os.path.basename(name))
            with open(full, "wb") as f:
                f.write(content)

        # write code to main.py
        main_path = os.path.join(td, "main.py")
        with open(main_path, "w", encoding="utf-8") as f:
            f.write(code_str)

        # run
        proc = subprocess.Popen(
            ["python", "main.py"],
            cwd=td,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        out, err = proc.communicate(timeout=120)
        return out.strip(), err.strip()
