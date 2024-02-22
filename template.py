import os
from pathlib import Path
import logging

#logging string
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'diffusion'

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/model.py", 
    f"src/{project_name}/components/train.py",
    f"src/{project_name}/components/dataloader.py",
    f"src/{project_name}/components/evaluate.py",
    f"src/{project_name}/components/sheduler.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/utils/logger.py",
    f"src/{project_name}/utils/exception.py", 
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/train_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    "research/trials.ipynb",
    "config/config.yaml",
    'data/.gitkeep', 
    "requirements.txt",
    "setup.py",
    "templates/index.html"
    'static/.gitkeep',
    'tests/__init__.py', 
    'README.md', 
    'LICENSE', 
    'app.py',
    'Dockerfile', 
    'artifact/.gitkeep',
    'artifact/model_ckpt/.gitkeep',
    'fig/.gitkeep', 
    'fig/img/.gitkeep',
    'logs/.gitkeep', 

]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")