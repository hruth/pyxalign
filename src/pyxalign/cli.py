"""Command-line interface for pyxalign."""

import argparse
import os
import sys
from pyxalign.autorunner.autorunner import AutorunnerPtychoV2
from pyxalign.autorunner.config import AutorunnerConfig


def main():
    """Entry point for the pyxalign-autorunner CLI command."""
    parser = argparse.ArgumentParser(
        description='Run the pyxalign autorunner for laminography/tomography alignment and reconstruction.'
    )
    parser.add_argument(
        '--config-file-path',
        type=str,
        default=None,
        help='Path to the configuration YAML file (optional)'
    )
    parser.add_argument(
        '--state-folder',
        type=str,
        default=None,
        help='Path to the state folder containing or to contain autorunner_state_file.yaml'
    )

    args = parser.parse_args()

    # Handle --state-folder argument if provided
    config_file_path = args.config_file_path
    if args.state_folder is not None:
        # Check if the folder exists
        if not os.path.exists(args.state_folder):
            print(f"Error: The folder '{args.state_folder}' does not exist.")
            sys.exit(1)

        if not os.path.isdir(args.state_folder):
            print(f"Error: '{args.state_folder}' is not a folder.")
            sys.exit(1)

        # Look for autorunner_state_file.yaml in the folder
        state_file_path = os.path.join(args.state_folder, "autorunner_state_file.yaml")

        if os.path.exists(state_file_path):
            # File exists, use it as input to AutorunnerPtychoV2
            config_file_path = state_file_path
        else:
            # File doesn't exist, create it
            config = AutorunnerConfig()
            config.state.state_folder = args.state_folder
            config.save_to_dict(state_file_path)
            config_file_path = state_file_path

    # Create and run the autorunner
    autorunner = AutorunnerPtychoV2(config_file_path)
    autorunner.run()


if __name__ == '__main__':
    main()
