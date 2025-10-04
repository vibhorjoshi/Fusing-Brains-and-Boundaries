from pathlib import Path
from typing import List


class Config:
	"""Project configuration with minimal defaults to run Steps 5–7."""

	# Output directories
	OUTPUT_DIR = "./outputs"  # String path for compatibility
	FIGURES_DIR = "./outputs/figures"
	MODELS_DIR = "./outputs/models"
	LOGS_DIR = "./outputs/logs"

	# Data
	DATA_DIR: Path = Path("./building_footprint_results/data")
	DEFAULT_STATE: str = "RhodeIsland"  # try smallest by default; will try to auto-detect
	TRAINING_STATES: List[str] = ["RhodeIsland", "Delaware", "Connecticut"]  # Default states for multi-state training

	# Data settings (used for synthetic generation)
	PATCH_SIZE: int = 256
	VALIDATION_SPLIT: float = 0.2
	BATCH_SIZE: int = 4
	MAX_PATCHES_PER_STATE: int = 1000  # Increased for better training
	PATCHES_PER_STATE: int = 500  # Default number of patches to extract per state

	# RL (DQN) hyperparameters
	RL_LEARNING_RATE: float = 1e-3
	RL_MEMORY_SIZE: int = 10_000
	RL_EPSILON_START: float = 1.0
	RL_EPSILON_END: float = 0.05
	RL_EPSILON_DECAY: float = 0.995
	RL_BATCH_SIZE: int = 32
	
	# Enhanced RL hyperparameters (PPO)
	RL_HIDDEN_DIM: int = 256
	RL_GAMMA: float = 0.99
	RL_WEIGHT_DECAY: float = 1e-5
	PPO_EPOCHS: int = 10
	PPO_CLIP: float = 0.2
	VALUE_COEF: float = 0.5
	ENTROPY_COEF: float = 0.01
	GAE_LAMBDA: float = 0.95
	IMAGE_FEATURE_DIM: int = 128

	# Visualization/plotting
	SAVE_FIGURE_DPI: int = 200

	# Training (Mask R-CNN)
	NUM_EPOCHS: int = 50
	LEARNING_RATE: float = 5e-4
	WEIGHT_DECAY: float = 1e-4
	NUM_WORKERS: int = 0  # set >0 if supported on your OS

	# Safety: limit training epochs on CPU to keep demo fast
	ALLOW_SLOW_TRAIN_ON_CPU: bool = False

