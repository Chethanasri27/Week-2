import os

# Data paths
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
META_DIR = "meta"
SIGNNAMES_CSV = os.path.join(META_DIR, "signnames.csv")

# Model paths
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "traffic_sign_model.h5")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "model_weights.h5")

# Image settings
IMG_SIZE = 30
IMG_CHANNELS = 3
NUM_CLASSES = 43

# Training settings
BATCH_SIZE = 32
EPOCHS = 15
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Random seed for reproducibility
SEED = 42
