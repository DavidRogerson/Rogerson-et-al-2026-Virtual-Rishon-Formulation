import numpy as np

from pydantic import BaseModel, Field
from typing import  Literal,  Optional,  Sequence

from tenpy.networks.site import FermionSite

from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Square

from tenpy.tools.misc import to_array
from tenpy.networks import mpo


from .lattice import LatticeGaugeTheoryLattice

from ..networks.site import VirtualU1RishonSite, BinaryEncoderSite
from ..networks.mpo import MPOCompress
from ..linalg.np_conserved import apply_permute_patch

from ..networks.mpo import MPOCompress
from ..linalg.np_conserved import apply_permute_patch

__all__ = ['QED3VRQubitEncoding', 'QED3VRQubitEncodingConfig']

class QED3VRQubitEncodingConfig(BaseModel):
    """
    Configuration for the QED3VR model.

    This configuration class defines the parameters required to initialize the QED3VR model.
    It includes parameters for the number of fermion flavors, rotor dimension, lattice size, and other model-specific settings.
    """
    model_name: str = "QED3VRQubitEncoding"
    # Model parameters
    Nf: int = Field(
        1,
        description="Number of fermion flavors in the model."
    )
    Nr: int = Field(
        4,
        description="Number of qubit rotor sites (2^Nr is truncation of gauge field)."
    )
    m: Sequence[float] = Field(
        ...,
        description="List of masses for each fermion flavor."
    )
    g: float = Field(
        1.0,
        description="Gauge coupling constant. Default is 1.0."
    )
    a: float = Field(
        1.0,
        description="Lattice spacing. Default is 1.0."
    )
    theta: float = Field(
        0.0,
        description="Phase factor for the theta term. Default is 0.0."
    )
    y_staggered_hopping: Optional[Sequence[float]] = Field(
        [[1,1],[-1,-1]],
        description="Staggered hopping amplitudes."
    )
    # Lattice parameters
    Lx: int = Field(
        2,
        description="Number of unit cells in the x-direction."
    )
    Ly: int = Field(
        2,
        description="Number of unit cells in the y-direction."
    )
    bc_MPS: Literal['finite', 'infinite'] = Field(
        'finite',
        description="Boundary condition for the MPS, either 'finite' or 'infinite'."
    )
    bc_x: Literal['open', 'periodic'] = Field(
        'open',
        description="Boundary condition in the x-direction, either 'open' or 'periodic'."
    )
    bc_y: Literal['open', 'periodic'] = Field(
        'open',
        description="Boundary condition in the y-direction, either 'open' or 'periodic'."
    )
    order: Optional[str] = Field(
        'default',
        description="Order of the lattice sites. Default is 'default'."
    )
    gauge_order: Literal['before_target_new', 'after_source'] = Field(
        'before_target_new',
        description="Gauge ordering in the MPS. Default is 'before_target_new'. Honestly nothing else should be used here"
    )
    charge_mod: Optional[Sequence[int]] = Field(
        None,
        description="Charge modulus for conserved charges. If None, one charge per unit cell is conserved."
    )

    compress_mpo: bool = Field(
        True,
        description="If MPO should be compressed up to numerical precision, added in v2"
    )

    version: Literal['v1'] = 'v1'
    """Version of the model configuration, used for compatibility."""

class QED3VRQubitEncoding(CouplingMPOModel):
    """
    QED3VRQubitEncoding: Quantum Electrodynamics in 2+1 Dimensions with Qubit Encoded Virtual Rishon Sites

    This class implements a two-dimensional lattice gauge theory (QED3) on a square lattice using the virtual rishon construction.
    It supports multiple fermion flavors and arbitrary rotor dimensions, and is designed for tensor network simulations (TeNPy).

    Key Features
    ------------
    - Supports arbitrary number of fermion flavors (`Nf`).
    - Rotor sites with configurable dimension (`DimR`), controlling gauge field truncation.
    - Flexible boundary conditions: open or periodic in the x and y directions.
    - Customizable mass, coupling constant, lattice spacing, and theta term.
    - Staggered hopping amplitudes via `y_staggered_hopping`.
    - Hamiltonian terms include mass, hopping, electric field, and magnetic (plaquette) interactions.
    - Efficient encoding of gauge degrees of freedom via virtual rishon construction.
    - Suitable for quantum simulation and tensor network methods.

    Model Parameters
    ----------------
    - Nf (int): Number of fermion flavors.
    - DimR (int): Rotor site dimension. Odd values induce a background E-field of 0.5.
    - m (list of float): Mass for each fermion flavor.
    - g (float): Gauge coupling constant (default: 1.0).
    - a (float): Lattice spacing (default: 1.0).
    - theta (float): Phase factor for the theta term (default: 0.0).
    - y_staggered_hopping (array-like): Staggered hopping amplitudes.
    - bc_MPS (str): MPS boundary condition ('finite' or 'infinite').
    - bc_x (str): Lattice boundary condition ('open' or 'periodic').
    - order, gauge_order, charge_mod (optional): Advanced lattice/gauge configuration.

    Methods
    -------
    - init_sites(model_params): Returns None (sites are initialized via `init_matter_gauge_sites`).
    - init_matter_gauge_sites(model_params): Initializes matter (fermion) and gauge (rotor) sites.
    - init_lattice(model_params): Constructs the full lattice including virtual rishon sites.
    - add_multi_coupling(...): Adds multi-site coupling terms to the Hamiltonian.
    - init_terms(model_params): Initializes all Hamiltonian terms (mass, hopping, electric, magnetic).
    - calc_term_mpos(tol_zero=1e-15): Builds MPO representations for each Hamiltonian term.
    - get_reference(): Returns BibTeX citation and notes for the model.
    - get_latex(): Returns LaTeX string for the Hamiltonian.
    - get_extra_default_measurements(): Returns list of extra measurement observables.

    References
    ----------
    - Meth et al., "Simulating Two-Dimensional Lattice Gauge Theories on a Qudit Quantum Computer", Nature Physics 21, 570–576 (2025).
    """
    config_cls = QED3VRQubitEncodingConfig
    default_lattice=Square

    def __init__(self, model_params):
        self.Nf = model_params['Nf']  # Number of fermion flavors
        self.Nr = model_params['Nr']
        self.DimR = 2**(self.Nr)
        self.compress_mpo = model_params['compress_mpo']
        super().__init__(model_params)
        self.total_term_mpo = self.calc_term_mpos()


    def init_sites(self, model_params):
        return None

    def init_matter_gauge_sites(self, model_params):
        """
        Initialize the matter and gauge sites for the model.
        """
        matter_sites = [FermionSite() for _ in range(self.Nf)]
        matter_names = [f'f{i+1}' for i in range(self.Nf)]
        gauge_sites = [BinaryEncoderSite((self.Nr-1)-i, filling=0.5) for i in range(self.Nr)]
        gauge_names = [f'r2^{i}' for i in range(self.Nr)]
        return matter_sites, matter_names, gauge_sites, gauge_names
    
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
            charge_mod = model_params['charge_mod'],
            gauge_order=model_params['gauge_order'],
        )
        return lat
    
    def add_multi_coupling(self, strength, ops, op_string=None, category=None, plus_hc=False, switchLR='middle_i', allow_inconsumerate=False):
        coupling_shape = self.lat.possible_multi_couplings(ops)[-1]
        strength = to_array(strength, coupling_shape, allow_incommensurate=allow_inconsumerate)
        return super().add_multi_coupling(strength, ops, op_string, category, plus_hc, switchLR)
    
    def init_terms(self, model_params):
        
        m = model_params['m']
        g = model_params['g']
        a = model_params['a']
        y_staggered_hopping = np.array(model_params["y_staggered_hopping"])


        for fl in range(self.Nf):
            self.add_onsite([[m[fl], -m[fl]], [-m[fl], m[fl]]], fl, 'N', 'mass')
        #hopping terms
        # for fl in range(0,self.Nf):
        #     self.add_multi_coupling(1/(2*a),  [('C', [0,0],fl), ('Ud', [0,0], self.Nf), ('Cd', [1,0], fl)], category='hopping', plus_hc=True)
        #     self.add_multi_coupling(1/(2*a)*y_staggered_hopping, [('C', [0,0],fl), ('Ud', [0,0], self.Nf+1), ('Cd', [0,1], fl)], category='hopping', plus_hc=True, allow_inconsumerate=True)



        for fl in range(0,self.Nf):
            for r in range(self.Nr):
                ##### cd Id Id ... Id Ud U U U U  ... C####
                #####           ....  r .... 
                terms = [('Cd', [0,0],fl)]+[('Ud', [0,0], self.Nf + self.Nr*0 +  r)] + [('U', [0,0], self.Nf + self.Nr*0 + k) for k in range(r+1, self.Nr)]+[('C', [1,0], fl)]
                print(terms)
                self.add_multi_coupling(1/(2*a),  terms , category=f'hopping_{fl}', plus_hc=True, allow_inconsumerate=True)
                terms = [('Cd', [0,0],fl)]+[('Ud', [0,0], self.Nf + self.Nr*1 +  r)] + [('U', [0,0], self.Nf + self.Nr*1 + k) for k in range(r+1, self.Nr)]+[('C', [0,1], fl)]
                print(terms)
                self.add_multi_coupling(1/(2*a)*y_staggered_hopping,  terms , category=f'hopping_{fl}', plus_hc=True, allow_inconsumerate=True)

        for coup,_ in enumerate(self.lat.simple_lattice.pairs['nearest_neighbors']):
            for r_1 in range(self.Nr):
                #Efield term
                for r_2 in range(self.Nr):
                    i = self.Nf + self.Nr*coup + r_1
                    j = self.Nf +  self.Nr*coup + r_2
                    if i == j:
                        self.add_onsite(a*(g**2)/2, i, 'E E', 'electric')
                    else:
                        self.add_coupling(a*(g**2)/2, i, 'E', j, 'E', dx=[0,0], category='electric')
        

        # magnetic field term
        # self.add_multi_coupling(-1/(2*a*g**2),
        #     [('U', [0,0], self.Nf), ('U', [1,0], self.Nf+1), ('Ud', [0,1], self.Nf), ('Ud', [0,0], self.Nf+1)],
        #     category='magnetic', plus_hc=True)
        
        for r1 in range(self.Nr):
            for r2 in range(self.Nr):
                for r3 in range(self.Nr):
                    for r4 in range(self.Nr):
                        terms = [('Ud', [0,0], self.Nf + self.Nr*0 +  r1)] + [('U', [0,0], self.Nf + self.Nr*0 + k) for k in range(r1+1, self.Nr)]
                        terms += [('Ud', [1,0], self.Nf + self.Nr*1 +  r2)] + [('U', [1,0], self.Nf + self.Nr*1 + k) for k in range(r2+1, self.Nr)]
                        terms += [('U', [0,1], self.Nf + self.Nr*0 +  r3)] + [('Ud', [0,1], self.Nf + self.Nr*0 + k) for k in range(r3+1, self.Nr)]
                        terms += [('U', [0,0], self.Nf + self.Nr*1 +  r4)] + [('Ud', [0,0], self.Nf + self.Nr*1 + k) for k in range(r4+1, self.Nr)]
                        self.add_multi_coupling(-1/(2*a*g**2),  terms , category='magnetic', plus_hc=True, allow_inconsumerate=True)
            

    # Add MPO Compression
    def calc_H_MPO(self, tol_zero=1e-15):        
            mpo = super().calc_H_MPO(tol_zero)
            if self.compress_mpo:
                apply_permute_patch()
                mpo = MPOCompress.from_MPO(mpo)
                mpo, trunc_err = mpo.compress({})
            return mpo 
        
    def calc_term_mpos(self, tol_zero=1.e-15):
        """
        Build MPO representations for each Hamiltonian term in the model.

        This routine iterates over all terms stored in self.coupling_terms and
        self.onsite_terms, removes prefactors with absolute value smaller than
        ``tol_zero`` (to keep the MPOs compact), and constructs an MPO for each
        remaining term using tenpy.networks.mpo.MPOGraph.

        Implementation notes
        - The method expects that each entry in the combined term dictionaries
          is a TermList/TermCollection (e.g. tenpy.networks.terms.TermList /
          tenpy.networks.terms.OnsiteTerms) that supports `.remove_zeros(tol)`
          and `.max_range()` methods.
        - For each term list `ct`, the MPOGraph is created via
          mpo.MPOGraph.from_terms((ct,), self.lat.mps_sites(), self.lat.bc_MPS)
          and then converted to an MPO with `.build_MPO()`.
        - Metadata from the original term list is preserved on the returned MPO:
          - `MPO.max_range` is set from `ct.max_range()`.
          - `MPO.explicit_plus_hc` is copied from `self.explicit_plus_hc`
            if present (defaults to False).

        Parameters
        ----------
        tol_zero : float, optional
            Threshold below which scalar prefactors are treated as zero and
            removed from the term lists before building MPOs. Default is 1e-15.

        Returns
        -------
        dict[str, tenpy.networks.mpo.MPO]
            A mapping from the term name (the keys of the combined term
            dictionaries) to the constructed MPO object for that term.
        """
        term_mpos = {}

        combined_terms = {}
        for key, ct in self.coupling_terms.items():
            try:
                combined_terms[key].append(ct)
            except KeyError:
                combined_terms[key] = [ct]
        for key, ct in self.onsite_terms.items():
            try:
                combined_terms[key].append(ct)
            except KeyError:
                combined_terms[key] = [ct]
    
        for name, cts in combined_terms.items():
            # remove tiny prefactors to keep MPOs compact
            max_range = 0
            for ct in cts:
                ct.remove_zeros(tol_zero)
                max_range = max(ct.max_range(), max_range)
            # build MPOGraph from this term (single-term tuple expected by from_terms)
            H_MPO_graph = mpo.MPOGraph.from_terms( cts, self.lat.mps_sites(), self.lat.bc_MPS)
            H_MPO = H_MPO_graph.build_MPO()
            # preserve metadata from the term list
            H_MPO.max_range = max_range
            H_MPO.explicit_plus_hc = getattr(self, "explicit_plus_hc", False)
            term_mpos[name] = H_MPO
        return term_mpos

    
    def get_extra_default_measurements(self):
        measurements = super().get_extra_default_measurements()
        measurements.extend([('Rogerson_et_al_2026_Virtual_Rishon_Formulation.models.QED3_qubit_encoding', mes) for mes in ['m_N', 'm_N_N_corr', 'm_terms_of_H', 'm_gauss_law', 'm_entanglement_hamiltonian']])
        return measurements



def m_N(results, psi, model, simulation):
    res = model.lat.mps2lat_values(psi.expectation_value('N'))
    results['N'] = res


def m_N_N_corr(results, psi, model, simulation):
    res = model.lat.mps2lat_values(psi.correlation_function('N', 'N'), axes=[0,1])
    results['N_N_corr'] = res


def m_gauss_law(results, psi, model, simulation):
    m = model
    Ns = m.lat.mps2lat_values(psi.expectation_value("N"))
    Ns[np.isnan(Ns)] = 0
    E_x = Ns[:,:,m.Nf:m.Nf + m.Nr].sum(axis=-1)
    E_y = Ns[:,:,m.Nf + m.Nr:].sum(axis=-1)
    E_matter = Ns[:,:,:m.Nf].sum(axis=-1)
    # Calc difference of E fields on incoming vs outgoing X links add difference of E field on in vs out y links and subtract the sum of all matter charges
    Gs = (E_x - np.roll(E_x, 1, axis=0)) + (E_y - np.roll(E_y, 1, axis=1)) - E_matter
    results['GaussLaw'] = Gs

def m_terms_of_H(results, psi, model, simulation):
    for t, mpoC in model.total_term_mpo.items():
        results[t] = mpoC.expectation_value(psi)


def m_entanglement_hamiltonian(results, psi, model, simulation):
    results["ent_ham"] = psi.entanglement_spectrum(True)[psi.L//2]