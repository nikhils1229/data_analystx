import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define the GPT model to be used
# gpt-4o is a great choice for its balance of intelligence, speed, and cost.
OPENAI_MODEL = "gpt-4o"

# This remains the same
DOCKER_IMAGE_NAME = "data-analyst-sandbox"
