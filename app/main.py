# app/main.py

"""
Entry point for the Data Analyst Agent (FastAPI version).
This file only exposes the FastAPI app so that Uvicorn can run it.
"""

from .agent import app  # Import the FastAPI app defined in agent.py
