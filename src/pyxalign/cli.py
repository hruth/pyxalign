"""Command-line interface for pyxalign."""

import argparse
from pyxalign.autorunner.autorunner import AutorunnerPtychoV2


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

    args = parser.parse_args()

    # Create and run the autorunner
    autorunner = AutorunnerPtychoV2(args.config_file_path)
    autorunner.run()


if __name__ == '__main__':
    main()
