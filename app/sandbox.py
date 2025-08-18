import docker
import os
import io

from . import config

def build_docker_image():
    """Builds the Docker image for the sandbox environment."""
    client = docker.from_env()
    print("🐳 Building Docker image for sandbox...")
    try:
        client.images.build(
            path=".",  # Assumes Dockerfile is in the root directory
            dockerfile="Dockerfile",
            tag=config.DOCKER_IMAGE_NAME,
            rm=True
        )
        print(f"✅ Docker image '{config.DOCKER_IMAGE_NAME}' built successfully.")
    except docker.errors.BuildError as e:
        print(f"❌ Docker build failed: {e}")
        raise

def run_code_in_sandbox(code: str, files: dict[str, bytes]) -> tuple[str, str]:
    """
    Runs Python code inside a secure Docker container.

    Args:
        code: The Python code string to execute.
        files: A dictionary of filenames and their byte content to be placed in the container.

    Returns:
        A tuple containing (stdout, stderr).
    """
    client = docker.from_env()
    
    # Create a temporary directory to mount into the container
    temp_dir = os.path.abspath("./temp_work")
    os.makedirs(temp_dir, exist_ok=True)

    for filename, content in files.items():
        with open(os.path.join(temp_dir, filename), "wb") as f:
            f.write(content)

    # Add the code to be executed as a file
    with open(os.path.join(temp_dir, "main.py"), "w") as f:
        f.write(code)

    try:
        container = client.containers.run(
            image=config.DOCKER_IMAGE_NAME,
            command="python main.py",
            volumes={temp_dir: {'bind': '/usr/src/app', 'mode': 'rw'}},
            working_dir="/usr/src/app",
            detach=True,
            remove=False  # Keep container for logs
        )
        # Wait for the container to finish, with a timeout
        result = container.wait(timeout=180) # 3-minute timeout
        
        stdout = container.logs(stdout=True, stderr=False).decode('utf-8').strip()
        stderr = container.logs(stdout=False, stderr=True).decode('utf-8').strip()

        print(f"📦 Sandbox stdout:\n---\n{stdout}\n---")
        if stderr:
            print(f"📦 Sandbox stderr:\n---\n{stderr}\n---")

        return stdout, stderr

    except Exception as e:
        print(f"❌ Sandbox execution failed: {e}")
        return "", str(e)
    finally:
        # Cleanup: stop and remove the container and the temp directory
        try:
            container.stop()
            container.remove()
        except NameError: # container was never created
             pass
        # Clean up temp files
        for filename in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, filename))
        os.rmdir(temp_dir)
