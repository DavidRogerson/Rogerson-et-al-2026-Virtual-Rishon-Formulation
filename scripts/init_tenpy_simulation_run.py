import argparse
import sys
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render SGE template with provided arguments.")
    parser.add_argument('--directory', default=None, help="Output file to write the rendered script. If not set, prints to stdout.")
    args = parser.parse_args()

    output = {}
    shutil.copyfile("scripts/run_tenpy_simulation.py", "workspace/" + args.directory + "/run_tenpy_simulation.py")

