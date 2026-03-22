import os

# TensorFlow optimization flag
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Audio config
SAMPLE_RATE = 22050
N_MFCC = 52

# Hugging Face model config
HF_REPO_ID = "sahatanmoyofficial/lstm_model.keras"
HF_MODEL_FILENAME = "lstm_model.keras"
