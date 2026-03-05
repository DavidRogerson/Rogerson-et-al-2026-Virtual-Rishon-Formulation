from pydantic import BaseModel, Field, model_validator
from typing import Union, Literal
from ..linalg.truncation import truncateConfig
import numpy as np


class AlgorithmConfig(BaseModel):
    """Base configuration for algorithms in TenPy."""
    max_N_sites_per_ring: Union[int, None] = Field(
        default=18,
        description="Maximum number of sites per ring in the algorithm."
    )
    trunc_params: truncateConfig = Field(...,
        description="Parameters for truncation in the algorithm."
    )
    version: Literal["v1"] = Field(
        default="v1",
        description="Version of the algorithm configuration."
    )