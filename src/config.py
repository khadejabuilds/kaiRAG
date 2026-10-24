from functools import lru_cache
from pathlib import Path
import yaml

@lru_cache
def get_config(path="configs/config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())