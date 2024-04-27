import yaml


def read_config() -> dict[str : str | dict[str:str]]:
    """Load config.yaml file and retrun it as dictionary.

    Returns:
        dict: Configuration loaded from YAML.
    """
    config = None
    with open("backend/config/config.yaml") as file:
        config = yaml.safe_load(file)

    return config
