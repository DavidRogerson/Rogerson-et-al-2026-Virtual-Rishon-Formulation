import numpy as np
from pydantic import BaseModel, Field
from typing import Union, Literal, Optional, Sequence

from tenpy.networks.site import FermionSite
from tenpy.models.model import CouplingMPOModel



from .lattice import LatticeGaugeTheoryLattice
from ..networks.site import VirtualU1RishonSite, BinaryEncoderSite
from ..networks.mpo import MPOCompress
from ..linalg.np_conserved import apply_permute_patch

__all__ = ['MassiveSchwingerModelQubitEncodingConfig', 'MassiveSchwingerModelQubitEncoding']

from pydantic import BaseModel, Field
from typing import Union, Sequence, Literal, Optional
class MassiveSchwingerModelQubitEncodingConfig(BaseModel):
    """
    Configuration for the MassiveSchwingerModelVR.
    
    This configuration class defines the parameters required to initialize the MassiveSchwingerModelVR.
    It includes parameters for the number of fermion flavors, rotor dimension, and other model-specific settings.
    """
    model_name: str = "MassiveSchwingerModelVR"
    #Model params
    Nf: int = Field(
        ...,
        ge=1,
        description="Number of fermion flavors in the model."
    )
    Nr: int = Field(
        ...,
        description="Number of qubits encoding the rotor. RodorDim=2^Nr"
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
    bc_x: Literal['open', 'periodic'] = Field(
        'open',
        description="Boundary condition in the x-direction, either 'open' or 'periodic'."
    )
    order: Literal['default', 'folded'] = Field(
        'default',
        description="Order of the lattice, either 'default' or 'folded'."
    )
    gauge_order: Literal['before_target', 'after_source'] = Field(
        'before_target',
        description="Where the gauge terms appear in the MPS ordering, either 'before_target' or 'after_source'."
    )

    charge_mod: Optional[Union[None, Sequence[int]]] = Field(
        None,
        description="Charge modolus, this defines the number individuall extensive conserved charges in the model. If None, one charges per unit cell is conserved, otherwise it will be charge_mod[i] indepdendent charges that alternate." 
    )

    compress_mpo: bool = Field(
        False,
        description="If MPO should be compressed up to numerical precision, added in v2"
    )

    version: Literal['v1', 'v2'] = 'v2'
    """Version of the model configuration, used for compatibility."""


class MassiveSchwingerModelQubitEncoding(CouplingMPOModel):
    """Massive Schwinger model with virtual rishon sites.
    This model extends the CouplingMPOModel to implement a massive Schwinger model
    constructed using virtual rishon sites. It initializes a lattice with a specified number of
    lattice sites, fermion flavors, and rotor dimensions. The model supports both open and periodic
    boundary conditions in the x-direction.
    For open boundary conditions, the there is a E-field on the left boundary. Which can be used to construct different boundary conditions.
    The E-field on the right boundary is assumed to be zero.
    For periodic boudnary conditions, the first and the last rotor are identified.
    Parameters
    ----------
    model_params : dict
        A dictionary containing the model parameters. It should include:
        - 'Nf': Number of fermion flavors (int).
        - 'DimR': Dimension of the rotor sites (int). Odd numbers result in a background E-field on the whole lattice of 0.5!
        - 'm': List of masses for each fermion flavor (list of floats).
        - 'g': Coupling constant (float, default is 1.0).
        - 'a': Lattice spacing (float, default is 1.0).
        - 'theta': Phase factor for the theta term (float, default is 0).
    Raises
    ------
    KeyError
        If required parameters are missing from `model_params`.
    ValueError
        If the parameters are not in the expected format or range.
    Notes
    -----
    - The model initializes a lattice with a specified number of sites and fermion flavors.
    - The rotor sites are initialized with a specified dimension and filling.
    - The model supports both open and periodic boundary conditions, affecting the initialization of the lattice.
    - The `init_sites` method initializes the sites for the model.
    - The `init_matter_gague_sites` method initializes the matter and gauge sites for the model.
    - The `init_lattice` method constructs the lattice with the initialized sites.
    - The `init_terms` method sets up the onsite and coupling terms for the model, including mass terms, hopping terms, gauge terms, and theta terms.
    - The `get_reference` method provides a reference for the model, including relevant papers and notes.
    Examples
    --------
    >>> model_params = {
    ...     'Nf': 2,
    ...     'DimR': 4,  # Odd numbers result in a background E-field on the whole lattice of 0.5!
    ...     'm': [1.0, 2.0],
    ...     'g': 1.0,
    ...     'a': 1.0,
    ...     'theta': 0.0,
    ...     'bc_MPS': 'finite',
    ...     'bc_x': 'open',
    ...     'order': 'default',
    ...     'order_gauge': 'before_target',
    ...     'charge_mod': None  # Optional, defaults to None
    ... }
    >>> model = MassiveSchwingerModelVR(model_params)
    """

    config_cls = MassiveSchwingerModelQubitEncodingConfig
    def __init__(self, model_params):
        self.Nf = model_params.get('Nf', 2)  # Number of fermion flavors
        self.Nr = model_params.get('Nr', 4)
        self.DimR = 2**self.Nr
        self.compress_mpo = model_params.get('compress_mpo', False)
        super().__init__(model_params)



    def init_sites(self, model_params):
        return [None]

    def init_matter_gauge_sites(self, model_params):
        """
        Initialize the matter and gauge sites for the model.
        """
        matter_sites = [FermionSite() for _ in range(self.Nf)]
        matter_names = [f'f{i+1}' for i in range(self.Nf)]

        gauge_sites = [BinaryEncoderSite((self.Nr-1)-i, filling=0.5) for i in range(self.Nr)]

        gauge_names = [f'r2^{i}' for i in range(self.Nr)]
        return matter_sites,matter_names, gauge_sites, gauge_names
    
    def init_lattice(self,model_params):
        simple_lat  = super().init_lattice(model_params)
        def U1RishonConstructor(sites_source, sites_target):
            """
            Constructor for the U(1) rishon site.
            """
            #return [VirtualU1RishonSite(s_s, s_t) for i,(s_s, s_t) in enumerate(zip(sites_source, sites_target))]
            return [VirtualU1RishonSite(s_s, s_t,offset=0.5 if i ==self.Nr-1 else 0) for i,(s_s, s_t) in enumerate(zip(sites_source, sites_target))]
        matter_sites, matter_names, gauge_sites, gauge_names = self.init_matter_gauge_sites(model_params)
        lat = LatticeGaugeTheoryLattice(
            simple_lat,
            matter_sites,
            gauge_sites,
            U1RishonConstructor,
            matter_names,
            gauge_names,
            remove_missing_bonds=True,
            new_charges='same',
            charge_mod = model_params.get('charge_mod', None),
            gauge_order=model_params.get('gauge_order', 'before_target'),
        )
        return lat
    
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
        for fl in range(0,self.Nf):
            for r in range(self.Nr):
                ##### cd Id Id ... Id Ud U U U U  ... C####
                #####           ....  r .... 
                terms = [('Cd', [0],fl)]+[('Ud', [0], r+self.Nf)] + [('U', [0], k+self.Nf) for k in range(r+1, self.Nr)]+[('C', [1], fl)]
                self.add_multi_coupling(-1.j/(2*a),  terms , category='hopping', plus_hc=True)
        for r in range(self.Nr):
            #theta term
            self.add_onsite(a*g**2*theta/(2*np.pi), self.Nf+r, 'E', 'theta')

        for r_1 in range(self.Nr):
            #gauge term
            for r_2 in range(self.Nr):
                if r_1 == r_2:
                    self.add_onsite(a*(g**2)/2, self.Nf+r_1, 'E E', 'gauge')
                else:
                    self.add_coupling(a*(g**2)/2, self.Nf+r_1, 'E', self.Nf+r_2, 'E', dx=[0], category='gauge')


    #def init_H_from_terms(self):
    #    pass

    
    def get_extra_default_measurements(self):
        measurements = super().get_extra_default_measurements()
        measurements.extend([('Rogerson_et_al_2026_Virtual_Rishon_Formulation.models.massive_schwinger_model_qubit_encoding', mes) for mes in ['m_chiral_condensate_E_field', 'm_correlation_chiral_condensate_E_field']])
        return measurements
    
    def calc_H_MPO(self, tol_zero=1e-15):        
        mpo = super().calc_H_MPO(tol_zero)
        if self.compress_mpo:
            apply_permute_patch()
            mpo = MPOCompress.from_MPO(mpo)
            mpo, trunc_err = mpo.compress({})
        return mpo 
    

def m_chiral_condensate_E_field(results, psi, model, simulation):
    res = model.lat.mps2lat_values(psi.expectation_value('N'))
    res[np.isnan(res)] = 0
    results['N'] = res
    results['E_field'] = res[...,model.Nf:].sum(axis=-1)
    results['E_field_total'] = np.sum(results['E_field'])
    for f in range(model.Nf):
        results[f'Gamma0_{f}'] = res[...,f]
        results[f'Gamma0_{f}_total'] = np.sum(res[...,f])
    results['Gausses_law'] = results['E_field'] -  np.roll(results['E_field'], 1) - res[:,:model.Nf].sum(axis=1)


def m_correlation_chiral_condensate_E_field(results, psi, model, simulation):
    res = model.lat.mps2lat_values(psi.correlation_function('N', 'N'), axes=[0,1])
    res_single = model.lat.mps2lat_values(psi.expectation_value('N'))
    results['N_N_corr'] = res
    for f, name in enumerate([f'Gamma0_{i}_Gamma0_{i}_corr' for i in range(model.Nf)]):
        results[name] = res[:,f,:,f] - np.outer(res_single[:,f], res_single[:,f])
    results['E_field_E_field_corr'] = np.sum(res[:,model.Nf:,:,model.Nf:], axis=(1,3)) - np.outer(res_single[:,model.Nf:].sum(axis=-1), res_single[:,model.Nf:].sum(axis=-1))