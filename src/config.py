"""
Configuration module for the Sarcasm Detection project.
Contains absolute paths and model hyperparameters used across the application.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "Sarcasm_Headlines_Dataset_v2.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sarcasm_model_final") 

BASE_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
LABEL_MAP = {0: "GENUINE", 1: "SARCASTIC"}