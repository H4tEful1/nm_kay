import pathlib
# from pathlib import Path
from pydantic import BaseModel, Field
import os


class Base(BaseModel):
    project_dir: str = os.path.abspath(os.path.join(os.path.abspath("nm_kay"), '..', '..'))
    database_path: str = project_dir + r'\database'

    # marked_mri_path: str = project_dir + r"\database\marked_mri"
    # processed_data_path: str = project_dir + r"\database\processed_data"
    # train_set_path: str = project_dir + r"\database\train_set"
    # path_to_model = project_dir + r"/best.pt"
    # prediction_path: str = project_dir + r"\database\predict"
    # morphometry_path: str = database_path + r"\for_test"
    # lic_path: str = database_path + r'\license.txt'

    class Config:
        env_file = f"{pathlib.Path(__file__).resolve().parent.parent}/.env"
        env_file_encoding = 'utf-8'


path_routing = Base()