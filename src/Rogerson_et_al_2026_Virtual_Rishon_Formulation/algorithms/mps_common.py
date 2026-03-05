from pydantic import BaseModel, Field, model_validator
from typing import Union, Literal
from ..linalg.krylov_based import LanczosGroundstateConfig
from .algorithm import AlgorithmConfig
import numpy as np



class MixerConfig(BaseModel):
    """Configuration for the mixer used in DMRG."""
    _default_amplitude = 1.e-5
    _default_decay = 2.
    _default_disable_after = 15

    amplitude: float = Field(
        default=_default_amplitude,
        description="Initial amplitude for the mixer (should be a real number)."
    )

    decay: Union[float,None] = Field(
        default=_default_decay,
        ge=1.0,
        description="Decay factor for the mixer (must be >= 1) or None."
    )

    disable_after: Union[int,None] = Field(
        default=_default_disable_after,
        gt=0,
        description="Number of sweeps after which the mixer is disabled. Must be > 0 or None if never."
    )

    version: Literal["v1"] = Field(
        default="v1",
        description="Version of the mixer configuration."
    )


class SweepConfig(AlgorithmConfig):
    """Configuration for the DMRG sweep."""
    chi_list: Union[dict,None] = Field(
        default=None,
        description="Bond dimension schedule for the sweep. If None, no schedule is used."
    )
    chi_list_reactivates_mixer: bool = Field(
        default=False,
        description="Whether the chi_list reactivates the mixer during the sweep."
    )
    combine: bool = Field(
        default=False,
        description="Whether to combine the left and right environments during the sweep."
    )
    lanczos_params: LanczosGroundstateConfig = Field(
        default_factory=LanczosGroundstateConfig,
        description="Parameters for the Lanczos ground state calculation."
    )
    #max_N_sites_per_ring from AlgorithmConfig is inherited
    mixer: Union[bool,str] = Field(
        True, description="Whether to use a mixer during the sweep."
    )
    mixer_params:MixerConfig = Field(
        default_factory=MixerConfig,
        description="Parameters for the mixer used in the sweep."
    )
    start_env: int = Field(
        default=1,
        description="Starting environment for the sweep. Must be >= 0."
    )

    version: Literal["v1"] = Field(
        default="v1",
        description="Version of the sweep configuration."
    )

class IterativeSweepConfig(SweepConfig):
    """Configuration for the iterative DMRG sweep."""
    max_hours: float = Field(
        default=24*365,
        description="Maximum number of hours to run the iterative sweep. If the time exceeds this value, the algorithm will stop."
    )
    max_sweeps: int = Field(
        default=1000,
        description="Maximum number of sweeps to perform in the iterative sweep. If the number of sweeps exceeds this value, the algorithm will stop."
    )
    max_trunc_err: Union[float,None] = Field(default=1e-4,
        description= "Threshold for raising errors on too large truncation errors. Default 0.0001. See consistency_check(). If the any truncation error eps on the final sweep exceeds this value, we raise. Can be downgraded to a warning by setting this option to None."
    )
    min_sweeps: int = Field(
        default=1,
        description="Minimum number of sweeps to perform in the iterative sweep. If the number of sweeps is below this value, the algorithm will not stop."
    )
