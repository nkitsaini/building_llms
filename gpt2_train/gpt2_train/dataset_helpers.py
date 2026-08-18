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

def get_named_dir(name: str) -> Path:
    d =  get_output_dir()/name
    d.mkdir(exist_ok=True)
    return d


def get_hellaswag_dir() -> Path:
    return get_named_dir('hellaswag')

def get_fineweb_dir() -> Path:
    return get_named_dir('fineweb_edu')

def get_checkpoint_dir() -> Path:
    return get_named_dir("checkpoint")

def get_log_filepath() -> Path:
    return get_named_dir("logs")/'logs.jsonl'
