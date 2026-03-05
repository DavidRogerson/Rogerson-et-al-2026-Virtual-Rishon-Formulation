import numpy as np
from pydantic import BaseModel, Field
from typing import Union,  Literal, Sequence

from tenpy.networks.site import  set_common_charges


from tenpy.models.model import CouplingMPOModel


from ..networks.site import StaggeredFermionSite
from ..networks.mpo import MPOCompress
from ..linalg.np_conserved import apply_permute_patch

from ..configs import types 

__all__ = ['MassiveSchwingerModelLongRangeConfig', 'MassiveSchwingerModelLongRange']

class MassiveSchwingerModelLongRangeConfig(BaseModel):
    """
    Configuration for the MassiveSchwingerModelLongRange.
    
    This configuration class defines the parameters required to initialize the MassiveSchwingerModelLongRange.
    It includes parameters for the number of fermion flavors, rotor dimension, and other model-specific settings.
    """
    model_name: str = "MassiveSchwingerModelLongRange"
    #Model params
    Nf: int = Field(
        ...,
        ge=1,
        description="Number of fermion flavors in the model."
    )
    m: Union[Sequence[float], Sequence[Sequence[float]]] = Field(
        ...,
        description="List of masses for each fermion flavor."
    )
    g: float = Field(
        1.0,
        description="Coupling constant of the gauge field. Default is 1.0."
    )
    a: float = Field(
        1.0,
        description="Lattice spacing. Default is 1.0."
    )
    theta: float = Field(
        0.0,
        description="Phase factor for the theta term. Default is 0."
    )
    m_corr: bool = Field(
        True,
        description="Whether to apply a mass correction. Default is True, see https://arxiv.org/pdf/2305.04437."
    )
    #Lattice params
    L: int = Field(
        ...,
        ge=1,
        description="Number of unit cells in the lattice. Total MPS size is L*(Nf+1) for bc_x='periodic' or L*(Nf+1)-1 for bc_x='open'."
    )

    bc_MPS: Literal['finite', 'infinite'] = Field(
        'finite',
        description="Boundary condition for the MPS, either 'finite' or 'infinite'."
    )
    bc_x: Literal['open'] = Field(
        'open',
        description="Boundary condition in the x-direction, only 'open'."
    )

    order: Literal['default'] = Field(
        'default',
        description="Order of the lattice, only 'default''."
    )

    compress_mpo: bool = Field(
        False,
        description="If MPO should be compressed up to numerical precision, added in v2"
    )

    version: Literal['v1', 'v2'] = 'v2'
    """Version of the model configuration, used for compatibility."""


class MassiveSchwingerModelLongRange(CouplingMPOModel):
    """
    Massive Schwinger model with long-range interactions (open boundary conditions only).

    This model extends CouplingMPOModel to implement the massive Schwinger model
    with all-to-all long-range gauge couplings. It supports multiple fermion flavors,
    configurable lattice size, and only open boundary conditions.

    For open boundary conditions, there is an E-field on the left boundary, which can be
    used to construct different boundary conditions. The E-field on the right boundary is
    assumed to be zero.

    Parameters
    ----------
    model_params : dict
    Dictionary containing model parameters:
        - 'Nf': int
        Number of fermion flavors.
        - 'm': list[float] or list[list[float]]
        Masses for each fermion flavor.
        - 'g': float, optional
        Coupling constant (default: 1.0).
        - 'a': float, optional
        Lattice spacing (default: 1.0).
        - 'theta': float, optional
        Phase factor for the theta term (default: 0.0).
        - 'L': int
        Number of unit cells in the lattice.
        - 'bc_MPS': {'finite', 'infinite'}, optional
        Boundary condition for the MPS (default: 'finite').
        - 'bc_x': {'open'}, optional
        Boundary condition in the x-direction (must be 'open').
        - 'order': str, optional
        Lattice order (default: 'default').

    Raises
    ------
    KeyError
    If required parameters are missing from `model_params`.
    ValueError
    If parameters are not in the expected format or range.

    Notes
    -----
    - The model initializes a lattice with the specified number of sites and fermion flavors.
    - Only open boundary conditions are supported; periodic boundaries are not implemented.
    - The `init_sites` method initializes the fermion sites.
    - The `init_terms` method sets up the onsite and coupling terms, including mass, hopping, gauge, and theta terms.
    - The `get_reference` method provides a reference for the model.

    Examples
    --------
    >>> model_params = {
    ...     'Nf': 2,
    ...     'm': [1.0, 2.0],
    ...     'g': 1.0,
    ...     'a': 1.0,
    ...     'theta': 0.0,
    ...     'L': 8,
    ...     'bc_MPS': 'finite',
    ...     'bc_x': 'open',
    ...     'order': 'default'
    ... }
    >>> model = MassiveSchwingerModelLongRange(model_params)
    """

    config_cls = MassiveSchwingerModelLongRangeConfig
    def __init__(self, model_params):
        self.Nf = model_params.get('Nf', 2)  # Number of fermion flavors
        self.compress_mpo = model_params.get('compress_mpo', False)
        super().__init__(model_params)


    def init_sites(self, model_params):
        sites = [StaggeredFermionSite() for _ in range(self.Nf)]
        set_common_charges(sites=sites, new_charges='independent', new_names=[f"N_{i}" for i in range(self.Nf)])
        return sites, [f"f{i}" for i in range(self.Nf)]
    
    def init_terms(self, model_params):
        m = model_params['m']
        g = model_params['g']
        a = model_params['a']
        theta = model_params['theta']
        m_corr = model_params['m_corr']
        if m_corr:
           m_corr = -self.Nf*g**2*a/8
        else:
           m_corr = 0
        #mass terms
        for fl in range(self.Nf):
            self.add_onsite([(m[fl]+m_corr), -(m[fl] + m_corr)], fl, 'N', 'mass')
        #hopping terms
        for fl in range(self.Nf):
            self.add_coupling(-1.j/(2*a), fl, 'Cd', fl, 'C', dx=[1],  category='hopping', plus_hc=True)
        #gauge term
        #needs to be redone !
        for i in range(self.lat.simple_lattice.N_sites-1):
            for j_1 in range(i+1):
                for f_1 in range(self.Nf):
                    for j_2 in range(i+1):
                        for f_2 in range(self.Nf):
                            op_1 = 'Q_even' if j_1%2 == 0 else 'Q_odd'
                            op_2 = 'Q_even' if j_2%2 == 0 else 'Q_odd'
                            mps_idx_1 = j_1*self.Nf+f_1
                            mps_idx_2 = j_2*self.Nf+f_2
                            if mps_idx_1 == mps_idx_2:
                                self.add_onsite_term(a*g**2/2, mps_idx_1, ' '.join([op_1, op_2]), category='gauge')
                            else:
                                op_1, op_2 =  (op_1, op_2) if mps_idx_1 < mps_idx_2 else (op_2, op_1)
                                mps_idx_1, mps_idx_2 = (mps_idx_1, mps_idx_2) if mps_idx_1 < mps_idx_2 else  (mps_idx_2, mps_idx_1)
                                self.add_coupling_term(a*g**2/2, mps_idx_1, mps_idx_2, op_1, op_2, category='gauge')
        
        for i in range(self.lat.simple_lattice.N_sites-1):
            for j in range(i+1):
                op = 'Q_even' if j%2 == 0 else 'Q_odd'
                for f in range(self.Nf):
                    mps_idx = j*self.Nf + f
                    self.add_onsite_term(a*g**2*theta/(2*np.pi),mps_idx, op, category='theta')
    
    def get_extra_default_measurements(self):
        measurements = super().get_extra_default_measurements()
        measurements.extend([('Rogerson_et_al_2026_Virtual_Rishon_Formulation.models.massive_schwinger_model_long_range', mes) for mes in ['m_pseudo_chiral_condensate', 'm_chiral_condensate_E_field', 'm_correlation_chiral_condensate', ]])
        return measurements
    
    def calc_H_MPO(self, tol_zero=1e-15):        
        mpo = super().calc_H_MPO(tol_zero)
        if self.compress_mpo:
            apply_permute_patch()
            mpo = MPOCompress.from_MPO(mpo)
            mpo, trunc_err = mpo.compress({})
        return mpo 


def m_pseudo_chiral_condensate(results, psi, model, simulation, results_key='Gamma5'):
    for f in range(model.Nf):
        res_f = []
        terms, s, _ = model.lat.possible_multi_couplings([('Cd', [0],f), ('C', [1], f)])
        for i,k in terms:
            res_f.append(1.j*(psi.expectation_value_term([('C', i), ('Cd', k)]) + psi.expectation_value_term([('C', i), ('Cd', k)])))
        results[results_key + f'_{f}'] = np.real_if_close(np.array(res_f))

def m_chiral_condensate_E_field(results, psi, model, simulation):
    res = model.lat.mps2lat_values(psi.expectation_value('N'))
    results['N'] = res
    print(res)
    results['E_field'] = np.cumsum(np.sum(res, axis=1) - np.array([(1-(-1)**n)/2 for n in range(res.shape[0])]))
    results['E_field_total'] = np.sum(results['E_field'][np.isnan(results['E_field']) == False])
    for f in range(model.Nf):
        results[f'Gamma0_{f}'] = res[...,f]
        results[f'Gamma0_{f}_total'] = np.sum(res[...,f])
    E_shift = np.roll(results['E_field'], 1)
    E_shift[0] = 0

    results['Gausses_law'] = results['E_field']-E_shift - res.sum(axis=1)

def m_correlation_chiral_condensate(results, psi, model, simulation):
    res = model.lat.mps2lat_values(psi.correlation_function('N', 'N'), axes=[0,1])
    res_single = model.lat.mps2lat_values(psi.expectation_value('N'))
    results['N_N_corr'] = res
    for f, name in enumerate([f'Gamma0_{i}_Gamma0_{i}_corr' for i in range(model.Nf)]):
        results[name] = res[:,f,:,f] - np.outer(res_single[:,f], res_single[:,f])




