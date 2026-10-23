import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "isl-secret-key-3f7a9c2b8e1d4f6a0c5e2b9d")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///isl_platform.db")
    DEBUG = os.getenv("DEBUG", "False") == "True"
    AUDIO_FOLDER = os.getenv("AUDIO_FOLDER", "static/audio")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    MODELS_LSTM_DIR = os.path.join(BASE_DIR, "models_LSTM")
    MODELS_MLP_DIR = os.path.join(BASE_DIR, "Models_MLP")

    HAND_LANDMARKER_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

    LSTM_MODEL_PATH = os.path.join(MODELS_LSTM_DIR, "isl_lstm_model_70.keras")
    WEBCAM_KINEMATIC_MODEL_PATH = os.path.join(MODELS_LSTM_DIR, "isl_webcam_kinematic_70.keras")
    KINEMATIC_MEAN_PATH = os.path.join(MODELS_LSTM_DIR, "kin_mean.npy")
    KINEMATIC_STD_PATH = os.path.join(MODELS_LSTM_DIR, "kin_std.npy")
    MLP_MODEL_PATH = os.path.join(MODELS_MLP_DIR, "isl_mlp_model_70_handaug.keras")
    LABEL_MAP_PATH = os.path.join(MODELS_LSTM_DIR, "label_map.json")
    LABEL_CLASSES_PATH = os.path.join(MODELS_LSTM_DIR, "label_classes.npy")
    MODEL_CONFIG_PATH = os.path.join(MODELS_LSTM_DIR, "model_config.json")
    MODEL_CONFIG_MLP_PATH = os.path.join(MODELS_MLP_DIR, "model_config_mlp.json")

    REFERENCE_SHAPES_PATH = os.path.join(MODELS_MLP_DIR, "reference_shapes_70.json")
    REFERENCE_LANDMARKS_PATH = os.path.join(MODELS_LSTM_DIR, "reference_landmarks_70.json")

    LSTM_MEAN_PATH = os.path.join(MODELS_LSTM_DIR, "X_mean.npy")
    LSTM_STD_PATH = os.path.join(MODELS_LSTM_DIR, "X_std.npy")
    MLP_MEAN_PATH = os.path.join(MODELS_MLP_DIR, "X_mlp_mean.npy")
    MLP_STD_PATH = os.path.join(MODELS_MLP_DIR, "X_mlp_std.npy")

    GESTURE_GIFS_DIR = os.path.join(BASE_DIR, "static", "gesture_gifs")

    SEQUENCE_LENGTH = 20
    NUM_FEATURES = 126
    NUM_CLASSES = 70

    SUPPORTED_LANGUAGES = {
        "English": "en",
        "Hindi": "hi",
        "Kannada": "kn",
        "Tamil": "ta",
        "Telugu": "te",
        "Malayalam": "ml",
        "Marathi": "mr",
        "Urdu": "ur",
        "Gujarati": "gu"
    }

    BADGE_THRESHOLD = 40