from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Config directory path
CONFIG_DIR = BASE_DIR / "config"

LABEL_DATABASE_FILE = CONFIG_DIR / "label_database.json"
PROMPT_FILE = CONFIG_DIR / "prompt_template.json"
SCHEMA_FILE = CONFIG_DIR / "schema.json"

# Output directory path
OUTPUT_DIR = BASE_DIR / "reports"
SAVE_CONTEXT_DIR = BASE_DIR / "reports" / "context_save"
BATCH_DIR = BASE_DIR / "reports" / "gpt_batch"
