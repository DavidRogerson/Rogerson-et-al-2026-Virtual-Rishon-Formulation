import numpy as np

from tenpy.networks.site import Site, FermionSite
import tenpy.linalg.np_conserved as npc

#U1 Rishon sites:
#Based on JupyterNotebook 25-06-06_ProjectedRishonModel.ipynb by D. Rogerson
from tenpy.linalg.charges import LegCharge, LegPipe


class StaggeredFermionSite(FermionSite):
    def __init__(self, conserve='N', filling=0.5, Q_gauss=0):
        super().__init__(conserve, filling)
        self.add_op("Q_even", self.get_op("N"))
        self.add_op("Q_odd", self.get_op("N") - self.get_op('Id'))

class RotorSite(Site):
    """
    Create a :class:`Site` used for the rotor model.
    Local states are ``0, 1, ..., Nmax``.
    Parameters
    ----------
    Nmax : int
        The maximum number of particles that can be accommodated in the rotor site.
    conserve : str
        Defines what is conserved, can be 'N', 'parity', or 'None'.
    filling : float
        The filling of the rotor site, which is used to define the number operator.
    Attributes
    ----------
    Nmax : int
        The maximum number of particles that can be accommodated in the rotor site.
    conserve : str
        Defines what is conserved, can be 'N', 'parity', or 'None'.
    filling : float
        The filling of the rotor site, which is used to define the number operator.
    leg : npc.LegCharge
        The leg (basis) information for the tensor, which is derived from the conservation law.
    ops : dict
        A dictionary containing the operators defined for the rotor site, including
        destruction (B), creation (Bd), number (N), number squared (NN),
        deviation from filling (dN), squared deviation from filling (dNdN), and parity (P).
    state_labels : dict
        A dictionary mapping the state labels to their corresponding quantum numbers.
    charge_to_JW_parity : np.ndarray
        An array representing the charge to Jordan-Wigner parity mapping, which is trivial in this case.
    Notes
    -----
    This class is designed to create a rotor site with a specified maximum number of particles,
    conservation law, and filling. It defines the operators that act on the rotor site and provides
    a way to represent the rotor site in terms of its local states and operators.
    The rotor site is used in quantum many-body simulations, particularly in models that involve
    rotational degrees of freedom. The operators defined in this site include the destruction operator (B),
    creation operator (Bd), number operator (N), number squared operator (NN), deviation from filling operator (dN),
    squared deviation from filling operator (dNdN), and parity operator (P).
    The local states are represented as strings from '0' to 'Nmax', and the operators are defined in a way
    that respects the conservation law specified by the `conserve` parameter.
    """
    def __init__(self, Nmax=1, conserve='N', filling=0.):
        if not conserve:
            conserve = 'None'
        if conserve not in ['N', 'parity', 'None']:
            raise ValueError("invalid `conserve`: " + repr(conserve))
        dim = Nmax + 1
        states = [str(n) for n in range(0, dim)]
        if dim < 2:
            raise ValueError("local dimension should be larger than 1....")
        B = np.zeros([dim, dim], dtype=np.float64)  # destruction/annihilation operator
        for n in range(1, dim):
            B[n - 1, n] = 1
        Bd = np.transpose(B)  # .conj() wouldn't do anything
        # Note: np.dot(Bd, B) has numerical roundoff errors of eps~=4.4e-16.
        Ndiag = np.arange(dim, dtype=np.float64)
        N = np.diag(Ndiag)
        NN = np.diag(Ndiag**2)
        dN = np.diag(Ndiag - filling)
        dNdN = np.diag((Ndiag - filling)**2)
        P = np.diag(1. - 2. * np.mod(Ndiag, 2))
        ops = dict(B=B, Bd=Bd, N=N, NN=NN, dN=dN, dNdN=dNdN, P=P)
        if conserve == 'N':
            chinfo = npc.ChargeInfo([1], ['N'])
            leg = npc.LegCharge.from_qflat(chinfo, range(dim))
        elif conserve == 'parity':
            chinfo = npc.ChargeInfo([2], ['parity_N'])
            leg = npc.LegCharge.from_qflat(chinfo, [i % 2 for i in range(dim)])
        else:
            leg = npc.LegCharge.from_trivial(dim)
        self.Nmax = Nmax
        self.conserve = conserve
        self.filling = filling
        Site.__init__(self, leg, states, sort_charge=True, **ops)
        self.state_labels['vac'] = self.state_labels['0']  # alias
        self.charge_to_JW_parity = np.array([0] * leg.chinfo.qnumber, int)  # trivial

    def __repr__(self):
        """Debug representation of self."""
        return "RotorSite({N:d}, {c!r}, {f:f})".format(N=self.Nmax,
                                                       c=self.conserve,
                                                       f=self.filling)


class VirtualU1RishonSite(Site):
    """Create a :class:`Site` used for the quantum link model.

    It allows to conserve a local U(1) symmetry of a lattice Gauge theory, using standard abelian charge conservation codes.
    This site represents a physical rotor defined by two virtual rotors of a quantum link model.
    This is a modification of the Idea presented in https://arxiv.org/pdf/1404.7439.
    It projects the two rishon sites onto the relevant total filling conserving sector, the filling is defined as the sum of the filling of the two rishon sites.
    It therfore represents a U1 rotor only.
    It then defines operators on the physical space based on projections of operators acting on the virtual sites.
    This class is designed to combine two RishonSites it therefore assumes a one dimensional geometry, That is Chain or Ring or Ladder.

    Parameters
    ----------
    RishonSite_1 : RishonSite
        The first rishon site to be combined.
    RishonSite_2 : RishonSite
        The second rishon site to be combined.
    Attributes
    ----------
    filling : float
        The total filling of the combined rishon sites. This is the total rishon occupation number
    proj : npc.Array
        The projector operator that combines the hilbert space of the two rishon sites, onto the subspace defined by 'filling'.
    leg : npc.LegCharge
        The leg (basis) information for the tensor, which is derived from the projector.
    Nmax : int
        The maximum number of particles that can be accommodated in the virtual rishon site.
    ops : dict
        A dictionary containing the operators defined for the virtual rishon site, including
        energy (E), creation (Ud), annihilation (U), and number (N) operators.
    state_labels : dict
        A dictionary mapping the state labels to their corresponding quantum numbers.
    charge_to_JW_parity : np.ndarray
        An array representing the charge to Jordan-Wigner parity mapping, which is trivial in this case.
    Notes
    -----
    This class is designed to work with two RishonSite objects that have the same conservation law
    and filling. It constructs a projector that combines the two sites and defines the operators
    that act on the virtual rishon site. The resulting site can be used in quantum link model simulations.
    The projector is constructed based on the filling of the two rishon sites, and it ensures that
    the resulting site respects the quantum link model structure. The operators defined in this site
    include the energy operator (E), creation operator (Ud), annihilation operator (U), and number operator (N).    
    """
    def __init__(self, RishonSite_1, RishonSite_2, offset = 0):
        assert RishonSite_1.conserve == RishonSite_2.conserve == 'N'
        self.RishonSites = [RishonSite_1, RishonSite_2]
        self.filling = int(RishonSite_1.filling + RishonSite_2.filling)
        self.proj = self.get_projector(RishonSite_1, RishonSite_2, self.filling)
        leg = self.proj.get_leg('p')
        self.leg = leg
        self.Nmax = leg.ind_len
        E =  0.5*(self.project_op(RishonSite_1.get_op('Id'), RishonSite_2.get_op('N'))
            - self.project_op(RishonSite_1.get_op('N'), RishonSite_2.get_op('Id'))
        ) +offset*self.project_op(RishonSite_1.get_op('Id'), RishonSite_2.get_op('Id'))
        U = self.project_op(RishonSite_1.get_op('B'), RishonSite_2.get_op('Bd'))
        ops = {'E':E, 'Ud':U.conj().transpose(['p', 'p*']), 'U':U, 'N':E}
        es = np.diag(E.to_ndarray())
        states = [str(int(e)) if isint else str(e) for e, isint in zip(es,  np.isclose(es, np.round(es)))]
        Site.__init__(self, leg, states, sort_charge=True, **ops)
        #self.state_labels['vac'] = self.state_labels['0']  # alias
        self.charge_to_JW_parity = np.array([0] * leg.chinfo.qnumber, int)  # trivial


    def create_B(self, site):
        """
        Creates a B tensor operator for quantum link model projectors.

        This function implements equation (18) from https://arxiv.org/pdf/1404.7439.
        This function constructs a tensor operator B that acts as a projector in the quantum link model,
        enforcing the correct filling of link rishons in their subspace.

        Parameters
        ----------
        site : Site
            A tenpy Site object containing the leg (basis) information for the tensor.

        Returns
        -------
        npc.Array
            A tensor operator B with shape (1, dim, dim, dim) where dim is the dimension of 
            the site's leg. The tensor has legs labeled ['wL', 'wR', 'p', 'p*'].

        Notes
        -----
        The resulting tensor has the following properties:
        - Creates diagonal entries of 1 along the physical dimensions
        - Includes trivial and non-trivial leg additions for proper quantum number handling
        - Is completely blocked according to the quantum number structure
        """
        l = site.leg
        B = npc.zeros([l, l.conj()], labels = ['p1', 'p1*'])
        #B = B.add_trivial_leg(1, 'wR', qconj=-1)
        B = B.add_leg(LegCharge.from_trivial(l.ind_len, l.chinfo, qconj=-1), i=0, axis=1, label='wR')
        _, B = B.as_completely_blocked()
        for i in range(l.ind_len):
            B[i,i,i] = 1
        return B.transpose(['p1', 'p1*', 'wR'])

    def create_C(self, site, filling):
        """
        Creates a C tensor operator for quantum link model projectors.

        This function implements equation (18) from https://arxiv.org/pdf/1404.7439, creating 
        a tensor operator that is part of the projector enforcing correct filling of link rishons 
        in their subspace.

        Parameters
        ----------
        site : Site
            A tenpy Site object containing the leg (basis) information for the tensor.
        filling : int
            The filling number specifying the occupation for the quantum link.

        Returns
        -------
        npc.Array
            A tensor operator C with shape (dim, 1, dim, dim) where dim is the dimension of 
            the site's leg. The tensor has legs labeled ['wL', 'wR', 'p', 'p*'].

        Notes
        -----
        The resulting tensor has the following properties:
        - Creates diagonal entries of 1 in reverse order along the physical dimensions
        - Includes trivial and non-trivial leg additions for proper quantum number handling
        - Is completely blocked according to the quantum number structure
        """
        l = site.leg
        C = npc.zeros([l, l.conj()], labels = ['p2', 'p2*'])
        C = C.add_leg(LegCharge.from_trivial(l.ind_len, l.chinfo, qconj=1), i=0, axis=0, label='wL')
        _, C = C.as_completely_blocked()
        for i in range(l.ind_len):
            if filling-i >= 0:
                C[filling-i,i,i] = 1
        return C.transpose(['p2', 'p2*', 'wL'])

    def get_projector(self, RishonSite_1, RishonSite_2, filling):
        """
        Constructs the projector operator for the quantum link model.
        This function decomposes the projector of  equation (18) from https://arxiv.org/pdf/1404.7439.
        Using SVD to find the basis of the combined rotor that enforces the correct filling of link rishons in their subspace.
        Parameters
        ----------
        RishonSite_1 : RishonSite
            The first rishon site to be combined.
        RishonSite_2 : RishonSite
            The second rishon site to be combined.
        filling : int
            The filling number specifying the occupation for the quantum link.
        Returns
        -------
        npc.Array
            The projector operator that combines the two rishon sites, ensuring that the resulting
            operator respects the quantum link model structure.
        Notes
        -----
        This function constructs the projector operator by first creating the B and C tensors for the two rishon sites.
        It then combines these tensors using a tensor dot product to form the projector. The resulting
        projector is then decomposed using singular value decomposition (SVD) to find the basis that enforces the correct filling.
        The SVD is performed on the combined legs of the projector, and the resulting singular vectors are returned as the projector.
        """
        B = self.create_B(RishonSite_1)
        C = self.create_C(RishonSite_2, filling)
        Proj = npc.tensordot(B, C, axes = ['wR', 'wL'])
        U, S, V = npc.svd(Proj.combine_legs([['p1', 'p2'],['p1*', 'p2*']]), cutoff=1e-15, inner_labels=['p*', 'p'])
        assert U == V.conj().transpose(['(p1.p2)', 'p*'])
        return V.split_legs()
    
    def project_op(self, op_1: npc.Array, op_2: npc.Array) -> npc.Array:
        """
        Projects the operators `op_1` and `op_2` onto the virtual rishon space.
        This function implements the projection of two operators onto the virtual rishon space
        defined by the rishon sites, ensuring that the resulting operator respects the
        quantum link model structure.
        Parameters
        ----------
        op_1 : npc.Array
            The first operator to be projected, which should have legs labeled ['p', 'p*'].
        op_2 : npc.Array
            The second operator to be projected, which should also have legs labeled ['p', 'p*'].
        Returns
        -------
        npc.Array
            The projected operator with legs labeled ['p', 'p*'], representing the combined effect
            of the two input operators in the virtual rishon space.
        Notes
        -----
        The function replaces the labels 'p' and 'p*' in `op_1` and `op_2` with 'p1', 'p1*' and 'p2', 'p2*'
        respectively, to distinguish between the two rishon sites. It then performs a series of
        tensor contractions to combine the operators according to the quantum link model structure.
        The final operator is returned with the labels ['p', 'p*'], which corresponds to the
        virtual rishon space defined by the two rishon sites.
        """
        op_1 = op_1.replace_labels(['p', 'p*'], ['p1', 'p1*'])
        op_2 = op_2.replace_labels(['p', 'p*'], ['p2', 'p2*'])
        op = npc.tensordot(self.proj, op_1, axes=['p1*', 'p1'])
        op = npc.tensordot(op, op_2, axes=['p2*', 'p2'])
        op = npc.tensordot(op, self.proj.conj(), [['p1*', 'p2*'], ['p1', 'p2']])
        return op.transpose(['p', 'p*'])

    def __repr__(self):

        """Debug representation of self."""
        return "VirtualU1RishonSite({N:d}, {c!r}, {f:f})".format(N=self.Nmax,
                                                      c=self.leg.chinfo,
                                                      f=self.filling)

#### End U1 Rishon sites 


### Binary Encoded U1 Rotor:

class BinaryEncoderSite(Site):
    """
    BinaryEncoderSite represents the nth digit of a binary-encoded U(1) rotor site.
    
    This site encodes the occupation number of a U(1) rotor in binary form, where each site corresponds to a single binary digit (bit) of the total occupation number. The class supports different conservation laws, such as particle number ('N'), parity, or none, and provides the appropriate local operators and charge structure for use in tensor network simulations.
    
    Parameters
    ----------
    n : int
        The index of the binary digit (bit) represented by this site.
    conserve : str, optional
        The conservation law to enforce. Can be 'N' (number), 'parity', or 'None'. Defaults to 'N'.
    sort_charge : bool, optional
        Whether to sort the basis states by charge. Defaults to True.
    filling : float, optional
        The filling of the site, used to define the number operator.
    
    Attributes
    ----------
    charge_to_JW_parity : np.ndarray
        An array representing the charge to Jordan-Wigner parity mapping, initialized as trivial in this case.
    
    Raises
    ------
    ValueError
        If an invalid value is provided for `conserve`.
    """

    def __init__(self, n, conserve='N', sort_charge=True, filling=0.5):
        self.filling=filling
        if not conserve:
            conserve = 'None'
        if conserve not in ['N', 'parity', 'None']:
            raise ValueError("invalid `conserve`: " + repr(conserve))
        X = [[0., 1.], [1., 0.]]
        Y = [[0., -1.j], [+1.j, 0.]]
        Z = [[1., 0.], [0., -1.]]
        N = [[0., 0.], [0, 2.0**n]]
        Bd = [[0., 1.], [0., 0.]]  # == Sx + i Sy
        B = [[0., 0.], [1., 0.]]  # == Sx - i Sy
        ops = dict(Z=Z, B=B, Bd=Bd, N=N)
        if conserve == 'N':
            chinfo = npc.ChargeInfo([1], [f'N'])
            leg = npc.LegCharge.from_qflat(chinfo, [0, 2**n])
        else:
            ops.update(X=X, Y=Y)
            if conserve == 'parity':
                chinfo = npc.ChargeInfo([2], ['parity'])
                leg = npc.LegCharge.from_qflat(chinfo, [0, (2**n)%2])  # charge depends on the position of the binary digit
            else:
                leg = npc.LegCharge.from_trivial(2)
        self.conserve = conserve
        # Specify Hermitian conjugates
        Site.__init__(self, leg, ['0', '1'], sort_charge=sort_charge, **ops)
        self.charge_to_JW_parity = np.array([0] * leg.chinfo.qnumber, int)  # trivial
###
