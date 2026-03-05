import os
import copy
from tenpy.simulations import Simulation
from tenpy.tools.params import asConfig
from tenpy.tools.misc import find_subclass, get_recursive, set_recursive
from tenpy.tools.hdf5_io import Hdf5Saver

from ..algorithms.dmrg import DMRGEngineConfig
import h5py
from pydantic import BaseModel, Field
from typing import Sequence, Union, Literal

class LatticeProductStateConfig(BaseModel):
    method: str = "lat_product_state"
    product_state: Union[Sequence[Sequence[str]], Sequence[Sequence[Sequence[str]]]]
    allow_incommensurate: bool = True
    version: str = "v1"

class DMRGSimulationRampChiConfig(BaseModel):
    simulation_class: str = "GroundStateSearch"
    sequential: dict = Field(default_factory=lambda: {"recursive_keys": ["algorithm_params.trunc_params.chi_max"]})
    output_filename: str = "results_simulation.h5"
    save_psi:bool = True
    model_class: str
    model_params: dict
    initial_state_params: LatticeProductStateConfig
    algorithm_class: Literal["TwoSiteDMRGEngine", "SingleSiteDMRGEngine"] = "TwoSiteDMRGEngine"
    algorithm_params: DMRGEngineConfig
    version: str = "v1"

class DMRGSimulationRampChiAndDiagMethodConfig(BaseModel):
    simulation_class: str = "GroundStateSearch"
    sequential: dict = Field(default_factory=lambda: {"recursive_keys": ["algorithm_params.trunc_params.chi_max", "algorithm_params.diag_method"]})
    output_filename: str = "results_simulation.h5"
    save_psi:bool = True
    model_class: str
    model_params: dict
    initial_state_params: LatticeProductStateConfig
    algorithm_class: Literal["TwoSiteDMRGEngine", "SingleSiteDMRGEngine"] = "TwoSiteDMRGEngine"
    algorithm_params: DMRGEngineConfig
    version: str = "v1"
