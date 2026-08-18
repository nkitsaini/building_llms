from pathlib import Path
import os

def get_output_dir() -> Path:
    if "DATASET_DIR" in os.environ:
        return Path(os.environ['DATASET_DIR'])

    if "__file__" in locals():
        output_dir = Path(__file__).parent.parent/'dataset'
    else:
        # in jupyter notebook
        output_dir = Path(os.getcwd()) / 'gpt2_train'/'dataset'
    return output_dir

def get_helloswap_d() -> Path:
    if "DATASET_DIR" in os.environ:
        return Path(os.environ['DATASET_DIR'])

    if "__file__" in locals():
        output_dir = Path(__file__).parent.parent/'dataset'
    else:
        # in jupyter notebook
        output_dir = Path(os.getcwd()) / 'gpt2_train'/'dataset'
    return output_dir