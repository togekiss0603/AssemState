import os
from pathlib import Path

# Get project root directory (assuming config.py is in the project root)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Define paths relative to project root
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = os.environ.get("ASSEMSTATE_DATA_ROOT", str(REPO_ROOT / "examples" / "data"))
SCENE_DIR = os.path.join(DATA_DIR, "prescene")
MULTIVIEW_SCENE_DIR = os.environ.get(
    "MULTIVIEW_SCENE_DIR",
    os.path.join(DATA_DIR, "prescene"),
)
MANUAL_DATA_PATH = os.path.join(DATA_DIR, "main_data.json")
OUTPUT_DIR = os.environ.get("ASSEMSTATE_OUTPUT_ROOT", str(REPO_ROOT / "outputs" / "tree"))

# Default model settings
DEFAULT_MODEL = "gpt-5"
DEFAULT_MAX_TOKENS = 120000
DEFAULT_TEMPERATURE = 0
