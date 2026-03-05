from pydantic import BaseModel, Field, model_validator
from typing import Union, Literal, Sequence
from ..configs import types
import numpy as np



class truncateConfig(BaseModel):
    """Configuration for truncation in TenPy."""
    chi_max: types.SequenceableInt = Field(100,
        description="Maximum bond dimension for truncation."
    )
    chi_min: types.SequenceableIntNone = Field(None,
        description="Minimum bond dimension for truncation. If None, no minimum is enforced."
    )
    degeneracy_tol:types.SequenceableFloatNone = Field(None,
        description="Tolerance for degeneracy in the truncation. If None, no degeneracy is conserved."
    )
    svd_min: types.SequenceableFloat = Field(
        default=float(np.finfo(np.complex128).eps * 100),
        description="Minimum singular value for truncation. Used to determine when to stop the SVD."
    )
    trunc_cut: types.SequenceableFloat = Field(
        default=float(np.finfo(np.complex128).eps * 100),
        description="Cutoff for truncation, used to determine when to stop the SVD."
    )
    version: Literal["v1"] = Field(
        default="v1",
        description="Version of the truncation configuration."
    )