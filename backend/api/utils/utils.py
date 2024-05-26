import yaml
from functools import lru_cache


@lru_cache
def read_config() -> dict[str : str | dict[str:str]]:
    """Load config.yaml file and retrun it as dictionary.

    Returns:
        dict: Configuration loaded from YAML.
    """
    config = None
    print("Loading config")
    with open("config/config.yaml") as file:
        config = yaml.safe_load(file)

    return config
