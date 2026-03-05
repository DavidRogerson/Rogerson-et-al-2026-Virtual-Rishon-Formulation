#Configuration for Krylov-based algorithms in TenPy
from pydantic import BaseModel, Field, model_validator
from ..configs import types
from typing import Union, Literal
import numpy as np

class KrylovBaseConfig(BaseModel):
    """Base configuration for Krylov-based algorithms."""
    cutoff: types.SequenceableFloat = Field(
        default=float(np.finfo(np.complex128).eps * 100),
        description="Cutoff for the Krylov algorithm, used to determine when to stop the iteration."
    )
    E_tol: types.SequenceableFloat = Field(
        default=100,
        description="Energy tolerance for the Krylov algorithm, used to determine when to stop the iteration."
    )
    
    E_shift: types.SequenceableFloatNone = Field(
        default=None,
        description="Energy shift for the Krylov algorithm, used to improve convergence."
    )
    min_gap: types.SequenceableFloat = Field(
        default=1e-12,
        description="Minimum gap for the Krylov algorithm, used to ensure that the eigenvalues are well-separated."
    )
    N_max: types.SequenceableInt = Field(
        default=20,
        description="Maximum number of Krylov iterations to perform."
    )
    N_min: types.SequenceableInt = Field(
        default=2,
        description="Minimum number of Krylov iterations to perform before stopping."
    )
    P_tol: types.SequenceableFloat = Field(
        default=1e-14,
        description="Tolerance for the Krylov algorithm, used to determine when to stop the iteration."
    )
    reortho: types.SequenceableBool = Field(
        default=False,
        description="Whether to reorthogonalize the Krylov vectors during the iteration."
    )
    version: Literal["v1"] = Field(
        default="v1",
        description="Version of the Krylov algorithm configuration."
    )

class LanczosGroundstateConfig(KrylovBaseConfig):
    """Configuration for Lanczos ground state calculation."""

    N_cache: types.SequenceableIntNone = Field(
        default=None,
        description="Number of Lanczos vectors to cache during the ground state calculation."
    )
    version: Literal["v1"] = Field(
        default="v1",
        description="Version of the Lanczos ground state configuration."
    )

    @model_validator(mode="after")
    def set_N_cache(self):
        if self.N_cache is None:
            self.N_cache = self.N_max
        return self