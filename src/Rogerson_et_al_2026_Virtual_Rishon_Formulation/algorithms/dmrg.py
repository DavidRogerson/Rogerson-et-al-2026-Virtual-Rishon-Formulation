from pydantic import BaseModel, Field, model_validator
from typing import Union, Literal
import numpy as np
from ..configs.types import SequenceableStr
from .mps_common import SweepConfig, MixerConfig, IterativeSweepConfig

class DMRGEngineConfig(IterativeSweepConfig):
    """Configuration for the DMRG engine."""
    # chi_list is inherited from SweepConfig
    #chi_list_reactivates_mixer inherited from SweepConfig
    #combine inherited from SweepConfig
    diag_method: SequenceableStr = Field(
        default="default",
        description="Method for diagonalization in the DMRG engine. Options: 'default', 'lanczos', 'arpack', 'ED_block', 'ED_all'"
    )
    E_tol_max: float = Field(
        default=1.e-4,
        description="See E_tol_to_trunc"
    )
    E_tol_min: float = Field(
        default=5e-16,
        description="See E_tol_to_trunc"
    )
    E_tol_to_trunc: Union[float,None] = Field(
        default=None,
        description="It’s reasonable to choose the Lanczos convergence criteria 'E_tol' not many magnitudes lower than the current truncation error. Therefore, if E_tol_to_trunc is not None, we update E_tol of lanczos_params to max_E_trunc*E_tol_to_trunc, restricted to the interval [E_tol_min, E_tol_max], where max_E_trunc is the maximal energy difference due to truncation right after each Lanczos optimization during the sweeps."
    )
    # lanczos_params inherited from SweepConfig
    max_E_err: float = Field(
        default=1e-8,
        description="Maximum error of the entanglement entropy per DRMG sweep. The engine will stop once the error is below this value."
    )
    # max_hours inherited from IterativeSweepConfig
    max_N_for_ED: int = Field(
        default=400,
        description="Maximum dimension for exact diagonalization in the DMRG engine. If the dimension exeeds this value, lanczos will be used instead."
    )
    max_S_err: float = Field(
        default=1e-5,
        description="Maximum error of the entanglement entropy per DRMG sweep. The engine will stop once the error is below this value."
    )
    #max_sweeps inherited from IterativeSweepConfig
    #max_trunc_err inherited from IterativeSweepConfig
    #min_sweeps inherited from IterativeSweepConfig

    #mixer inherited from SweepConfig
    #mixer_params inherited from SweepConfig
    N_sweeps_check: int = Field(
        default=1,
        description="Number of sweeps to perform between checking convergence criteria and giving a status update."
    )
    norm_tol: float = Field(
        default=1e-5,
        description="After the DMRG run, update the environment with at most norm_tol_iter sweeps until np.linalg.norm(psi.norm_err()) < norm_tol."
    )
    norm_tol_iter: int = Field(
        default=5,
        description="Perform at most norm_tol_iter`*`update_env sweeps to converge the norm error below norm_tol."
    )
    norm_tol_final: float = Field(
        default=1e-10,
        description="After performing norm_tol_iter`*`update_env sweeps, if np.linalg.norm(psi.norm_err()) < norm_tol_final, call canonical_form() to canonicalize instead. This tolerance should be stricter than norm_tol to ensure canonical form even if DMRG cannot fully converge."
    )
    P_tol_max: float = Field(
        default=1e-4,
        description="See P_tol_to_trunc"
    )
    P_tol_to_trunc: float = Field(
        default=0.05,
        description="It’s reasonable to choose the Lanczos convergence criteria 'P_tol' not many magnitudes lower than the current truncation error. Therefore, if P_tol_to_trunc is not None, we update P_tol of lanczos_params to max_trunc_err*P_tol_to_trunc, restricted to the interval [P_tol_min, P_tol_max], where max_trunc_err is the maximal truncation error (discarded weight of the Schmidt values) due to truncation right after each Lanczos optimization during the sweeps."
    )
    P_tol_min: float = Field(
        default=1e-30,
        description="See P_tol_to_trunc"
    )
    #start_env inherited from SweepConfig
    #trunc_params inherited from AlgorithmConfig
    update_env: int = Field(
        default=5,
        description="How often many update environment sweeps should be performed."
    )
    version: Literal["v1"] = Field(
        default="v1",
        description="Version of the DMRG engine configuration."
    )



class iDMRGEngineConfig(DMRGEngineConfig):
    """Configuration for the iDMRG engine."""
    N_sweeps_check: int = Field(
        default=10,
        description="Number of sweeps to perform between checking convergence criteria and giving a status update."
    )
    min_sweeps: int = Field(
        default=15,
        description="Minimum number of sweeps to perform before checking convergence criteria."
    )
    mixer_params: MixerConfig = MixerConfig(disable_after = 50, decay = 2**(15/50))

    version: Literal["v1"] = Field(
        default="v1",
        description="Version of the iDMRG engine configuration."
    )