
import argparse
import sys

import tenpy as tp
import numpy as np
import scipy as sp
import Rogerson_et_al_2026_Virtual_Rishon_Formulation.models as models_ext
from tenpy.simulations import run_simulation, run_seq_simulations

from pprint import pprint
import json
import yaml
from pathlib import Path

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run a tenpy simulation reading a signac_statepoint as the input config")
    parser.add_argument('--config_file', required=True, help="Path to the configuration file for the tenpy simulation.")
    args = parser.parse_args()
    try:
        with open(args.config_file, 'r') as f:
            simulation_parameters = json.load(f)
        results = run_seq_simulations(**simulation_parameters)
        tp.tools.hdf5_io.save(results, simulation_parameters['output_filename'])
        with open('job_completed', 'w') as f:
            f.write("Simulation completed successfully.")
    except Exception as e:
        with open('job_error', 'w') as f:
            f.write(f"Simulation failed with error: {str(e)}")
        raise e

    