from pathlib import Path

ROOT = Path(__file__).parent.resolve().parent

PROJECT_DIR = ROOT
DATA_DIR = PROJECT_DIR / "datasets"
EMBEDDING_DIR = PROJECT_DIR / "train" / "scripts" / "embeddings"
# CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_DIR = PROJECT_DIR / "train" / "training_output"
FIGURE_DIR = PROJECT_DIR / "figures"
WB_DIR = PROJECT_DIR / "wandb"
