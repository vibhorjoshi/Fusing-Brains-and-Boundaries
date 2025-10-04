import os
from pathlib import Path

class Config:
    """Configuration class for the building footprint extraction pipeline."""
    
    # Base directories
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs"  # ✅ CRITICAL: This was missing!
    
    # Output subdirectories
    MODELS_DIR = OUTPUT_DIR / "models"
    FIGURES_DIR = OUTPUT_DIR / "figures"
    LOGS_DIR = OUTPUT_DIR / "logs"
    RESULTS_DIR = OUTPUT_DIR / "results"
    
    # Building footprint data
    BUILDING_DATA_DIR = BASE_DIR / "building_footprint_results" / "data"
    
    # Model parameters
    MASK_RCNN_BACKBONE = "resnet50"
    MASK_RCNN_NUM_CLASSES = 2  # background + building
    
    # Training parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Image parameters
    IMAGE_SIZE = 256
    PATCH_SIZE = 256
    
    # GPU settings
    USE_GPU = True
    DEVICE = "cuda" if USE_GPU else "cpu"
    NUM_WORKERS = 4
    
    # Regularization parameters
    RT_KERNEL_SIZE = 3
    RR_KERNEL_SIZE = 5
    FER_KERNEL_SIZE = 3
    
    # DQN parameters
    DQN_STATE_SIZE = 128
    DQN_ACTION_SIZE = 4  # Number of regularization combinations
    DQN_HIDDEN_SIZE = 256
    DQN_GAMMA = 0.99
    DQN_EPSILON_START = 1.0
    DQN_EPSILON_END = 0.01
    DQN_EPSILON_DECAY = 0.995
    
    # Evaluation
    IOU_THRESHOLD = 0.5
    
    # Legacy compatibility attributes
    DEFAULT_STATE = "RhodeIsland"
    TRAINING_STATES = ["RhodeIsland", "Delaware", "Connecticut"]
    VALIDATION_SPLIT = 0.2
    MAX_PATCHES_PER_STATE = 1000
    PATCHES_PER_STATE = 500
    RL_LEARNING_RATE = 1e-3
    RL_MEMORY_SIZE = 10_000
    RL_EPSILON_START = 1.0
    RL_EPSILON_END = 0.05
    RL_EPSILON_DECAY = 0.995
    RL_BATCH_SIZE = 32
    RL_HIDDEN_DIM = 256
    RL_GAMMA = 0.99
    RL_WEIGHT_DECAY = 1e-5
    PPO_EPOCHS = 10
    PPO_CLIP = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    GAE_LAMBDA = 0.95
    IMAGE_FEATURE_DIM = 128
    SAVE_FIGURE_DPI = 200
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 0
    ALLOW_SLOW_TRAIN_ON_CPU = False
    
    @classmethod
    def create_directories(cls):
        """Create all necessary output directories."""
        directories = [
            cls.OUTPUT_DIR,
            cls.MODELS_DIR,
            cls.FIGURES_DIR,
            cls.LOGS_DIR,
            cls.RESULTS_DIR,
            cls.DATA_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    @classmethod
    def get_state_data_path(cls, state_name):
        """Get the path to state-specific building footprint data."""
        return cls.BUILDING_DATA_DIR / state_name

