from os.path import join as pjoin
import os.path as osp

from PIL import Image

from src.utils import *

SEKAI_DATA_DIR = "data/pjsk"
SEKAI_CONFIG_DIR = "config/pjsk"
SEKAI_ASSET_DIR = f"{SEKAI_DATA_DIR}/assets"

config = Config("pjsk.pjsk")
logger = get_logger("pjsk")
file_db = get_file_db(f"{SEKAI_DATA_DIR}/db.json")

ALL_SERVER_REGIONS = ["jp", "tw", "cn"]
ALL_SERVER_REGION_NAMES = ["日服", "台服", "国服"]

try:
    UNKNOWN_IMG = Image.open(f"{SEKAI_ASSET_DIR}/static_images/unknown.png")
except Exception:
    UNKNOWN_IMG = None