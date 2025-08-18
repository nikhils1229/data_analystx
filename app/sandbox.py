import subprocess
import os
import tempfile

def run_code_in_sandbox(code: str, files: dict[str, bytes]) -> tuple[str, str]:
    """
    Runs Python code inside a secure subprocess.

    Args:
        code: The Python code string to execute.
        files: A dictionary of filenames and their byte content to be placed in the sandbox.

    Returns:
        A tuple containing (stdout, stderr).
    """
    # Create a temporary directory to act as the sandbox
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write any provided files to the temporary directory
        for filename, content in files.items():
            with open(os.path.join(temp_dir, filename), "wb") as f:
                f.write(content)

        # The path to the script to be executed
        script_path = os.path.join(temp_dir, "main.py")

        # Write the AI-generated code to a file
        with open(script_path, "w") as f:
            f.write(code)

        try:
            # Execute the script in a new process
            # We use cwd (current working directory) to ensure the script can find the files
            process = subprocess.run(
                ["python", "main.py"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=180  # 3-minute timeout
            )
            
            stdout = process.stdout
            stderr = process.stderr

            print(f"📦 Subprocess stdout:\n---\n{stdout}\n---")
            if stderr:
                print(f"📦 Subprocess stderr:\n---\n{stderr}\n---")
                
            return stdout, stderr

        except subprocess.TimeoutExpired:
            return "", "Error: Code execution timed out after 3 minutes."
        except Exception as e:
            print(f"❌ Subprocess execution failed: {e}")
            return "", str(e)
