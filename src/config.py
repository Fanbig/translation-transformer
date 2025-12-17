from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# 路径
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

# 训练参数
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50

# 模型结构
DIM_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
