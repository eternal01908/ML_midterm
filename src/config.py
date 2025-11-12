# src/config.py
# File chứa các hằng số và cấu hình chung của dự án

import os

# --- Hằng số Mô hình ---
RANDOM_STATE = 42
TEST_SIZE = 0.3

# --- Hằng số Dữ liệu ---
KAGGLE_DATASET = "sgcongb/youtube-revenue-data-2018-2021"
CSV_FILES_TO_LOAD = ['Table data 2018.csv', 'Table data 2019.csv', 'Table data 2020.csv']

# --- Đường dẫn (Paths) ---
# Tự động lấy đường dẫn gốc của dự án
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'dataset_clean.csv')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')