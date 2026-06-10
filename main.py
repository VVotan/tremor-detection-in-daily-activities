import argparse
from pathlib import Path

import yaml


def parse_arguments()-> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Tremor Frequency Analysis"
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help=(
            "Path to YAML configuration file. "
            "Example: configs/example_config.yaml"
        ),
    )

    return parser.parse_args()

def load_config(config_file: Path) -> dict:
    """
    Load YAML configuration.

    Parameters
    ----------
    config_file : Path to YAML file.

    -------
    dict : Parsed configuration.
    """

    with open(config_file, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main() -> None:
    """
    Application entry point.

    Loads the YAML configuration and starts the analysis
    pipeline.
    """

    args = parse_arguments()

    config = load_config(args.config)
    pass
    #TODO: Run our analysis


if __name__ == "__main__":
    main()