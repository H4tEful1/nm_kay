import pathlib
# from pathlib import Path
from pydantic import BaseModel, Field
import os


class Base(BaseModel):
    project_dir: str = os.path.abspath(os.path.join(os.path.abspath("nm_kay"), '..', '..'))
    database_path: str = project_dir + r'\database'

    class Config:
        env_file = f"{pathlib.Path(__file__).resolve().parent.parent}/.env"
        env_file_encoding = 'utf-8'


path_routing = Base()