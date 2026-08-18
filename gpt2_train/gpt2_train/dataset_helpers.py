from pathlib import Path
import os

def get_output_dir() -> Path:
    if "DATASET_DIR" in os.environ:
        return Path(os.environ['DATASET_DIR'])

    if "__file__" in globals():
        output_dir = Path(__file__).parent.parent/'dataset'
    else:
        # in jupyter notebook
        output_dir = Path(os.getcwd()) / 'gpt2_train'/'dataset'
    assert output_dir.exists(), f"Expected {output_dir} to be present"
    return output_dir

def get_hellaswag_dir() -> Path:
    d =  get_output_dir()/'hellaswag'
    d.mkdir(exist_ok=True)
    return d

def get_fineweb_dir() -> Path:
    d = get_output_dir()/'fineweb_edu'
    d.mkdir(exist_ok=True)
    return d
