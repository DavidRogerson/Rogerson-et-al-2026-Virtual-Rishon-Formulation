import numpy as np
from tenpy.tools.params import asConfig
from tenpy.networks.mpo import MPO
import tenpy.linalg.np_conserved as npc
from tenpy.tools.params import asConfig
from tenpy.algorithms.truncation import TruncationError, svd_theta
import warnings
from numpy.exceptions import ComplexWarning

class MPOCompress(MPO):
    def __init__(self, sites, Ws, bc='finite', IdL=None, IdR=None, max_range=None, explicit_plus_hc=False):
        super().__init__(sites, Ws, bc, IdL, IdR, max_range, explicit_plus_hc)
    
    @classmethod
    def from_MPO(self, mpo):
        """
        Creates an MPOCompress instance from an existing MPO object.

        Parameters
        ----------
        mpo : MPO
            The MPO (Matrix Product Operator) object to create the compressed version from.
            Should contain the following attributes:
            - sites: List of sites
            - _W: MPO tensors
            - bc: Boundary conditions
            - IdL: Left identity matrix 
            - IdR: Right identity matrix
            - max_range: Maximum range of interactions
            - explicit_plus_hc: Boolean flag for Hermitian conjugate

        Returns
        -------
        MPOCompress
            A new MPOCompress instance initialized with the attributes from the input MPO.
        """
        return MPOCompress(mpo.sites, mpo._W, mpo.bc, mpo.IdL, mpo.IdR, mpo.max_range, mpo.explicit_plus_hc)


    def sort_legcharges(self, return_perms = False):
        """Sort virtual legs by charges. In place.

        The MPO seen as matrix of the ``wL, wR`` legs is usually very sparse. This sparsity is
        captured by the LegCharges for these bonds not being sorted and bunched. This requires a
        tensordot to do more block-multiplications with smaller blocks. This is in general faster
        for large blocks, but might lead to a larger overhead for small blocks. Therefore, this
        function allows to sort the virtual legs by charges.
        """
        new_W = [None] * self.L
        perms = [None] * (self.L + 1)
        for i, w in enumerate(self._W):
            w = w.transpose(['wL', 'wR', 'p', 'p*'])
            p, w = w.sort_legcharge([True, True, False, False], [True, True, False, False])
            if perms[i] is not None:
                assert np.all(p[0] == perms[i])
            perms[i] = p[0]
            perms[i + 1] = p[1]
            new_W[i] = w
        self._W = new_W
        chi = self.chi
        for b, p in enumerate(perms):
            IdL = self.IdL[b]
            if IdL is not None:
                self.IdL[b] = np.nonzero(p == IdL)[0][0]
            IdR = self.IdR[b]
            if IdR is not None:
                IdR = IdR % chi[b]
                self.IdR[b] = np.nonzero(p == IdR)[0][0]
        if return_perms:
            return perms

    def mirror(self):
        """
        Creates a mirrored version of the current MPO (Matrix Product Operator).
        This method reverses the order of the tensors and their internal structure to create
        a mirror image of the original MPO. The mirroring process includes:
        1. Reversing the order of the tensors
        2. Swapping the left and right indices
        3. Reversing the row and column permutations
        4. Adjusting the identity matrices positions
        Returns
        -------
        MPOCompress
            A new MPOCompress object representing the mirrored operator with:
            - Reversed site order
            - Transformed tensor elements (Ws)
            - Reversed boundary conditions
            - Adjusted identity matrices positions (IdLs, IdRs)
            - Same max_range and explicit_plus_hc parameters as original
        Note
        ----
        The mirroring operation preserves the physical meaning of the operator while
        reversing its spatial orientation.
        """
        Ws = []
        IdLs = []
        IdRs = []
        for i in range(self.L-1, -1, -1):
            W = self.get_W(i)
            W = W.replace_labels(['wL', 'wR'], ['wR', 'wL'])
            W.itranspose(['wL', 'wR', 'p', 'p*'])
            full_shape = W.shape
            row_permute = np.array([i for i in range(full_shape[0]-1, -1, -1)])
            col_permute = np.array([i for i in range(full_shape[1]-1, -1, -1)])
            W = W[row_permute, col_permute, : , :]
            Ws.append(W)
            if i == self.L-1:
                IdRs.append(row_permute[self.get_IdL(i+1)])
            
            IdLs.append(row_permute[self.get_IdR(i)])
            IdRs.append(col_permute[self.get_IdL(i)])
            
            if i == 0:
                IdLs.append(col_permute[self.get_IdR(i-1)])
        return MPOCompress(self.sites[::-1], Ws, self.bc, IdLs, IdRs, self.max_range, self.explicit_plus_hc)
    
    def left_canonical_form(self, *args, **kwargs):
        assert self.bc == "finite"
        return self._left_canonical_form_finite(*args, **kwargs)
    

    def _left_canonical_form_finite(self, absorb_R=False):
        """
        Transforms MPO into left canonical form for finite systems.

        This method performs a left-canonical decomposition of the MPO using QR decomposition,
        storing the R matrices and generating a new MPO in left-canonical form.

        Parameters
        ----------
        absorb_R : bool, optional
            If True, absorbs the final R matrix into the last site tensor.
            Default is False.

        Returns
        -------
        tuple
            A tuple containing:
            - mpo : MPOCompress
                The MPO in left-canonical form
            - Rs : list of Array
                The list of R matrices from the QR decompositions

        Notes
        -----
        The method proceeds site by site from left to right, performing QR decompositions
        and updating the identity matrices (IdL, IdR) that encode the operator structure.
        The resulting MPO has all tensors in left-canonical form except possibly the
        rightmost one if absorb_R is False.
        """
        Ws = []
        R = npc.diag(np.ones(self.get_W(0).shape[0]), self.get_W(0).get_leg('wL'), labels=['wL', 'wR'], dtype=self.dtype)
        Rs = [R]
        IdLs = self.IdL.copy()
        IdRs = self.IdR.copy()
        for i in range(self.L):
            A = npc.tensordot(R, self.get_W(i), axes=['wR', 'wL'])
            IdL = IdLs[i]
            IdLnext = IdLs[i+1]
            IdR = IdRs[i+1]
            IdRlast = IdRs[i]
            W, R, IdLnext, IdR = self._block_QR(A, IdL, IdLnext, IdRlast, IdR)
            IdLs[i+1] = IdLnext
            IdRs[i+1] = IdR
            Rs.append(R.copy())
            Ws.append(W.copy())
        if absorb_R:
            Ws[-1] = npc.tensordot(Ws[-1],R, axes=['wR', 'wL']).transpose(['wL', 'wR', 'p', 'p*'])
            IdRs[-1] = self.IdR[-1]
            IdLs[-1]= self.IdL[-1]
        mpo = MPOCompress(self.sites, Ws, self.bc, IdLs, IdRs, self.max_range, self.explicit_plus_hc)
        perms = mpo.sort_legcharges(return_perms=True)
        Rs = [R[perms[i], perms[i]] for i,R in enumerate(Rs)]
        return mpo, Rs

    def _compress_finite(self, trunc_params):
        """
        Compresses a finite Matrix Product Operator (MPO) using SVD decomposition.

        This method performs compression of a finite MPO by:
        1. Converting to right canonical form
        2. Performing sequential SVD decompositions from left to right
        3. Applying truncation according to given parameters

        Parameters
        ----------
        trunc_params : dict
            Dictionary containing truncation parameters for SVD compression.
            Expected keys:
            - 'chi_max': Maximum bond dimension
            - 'svd_min': Minimum singular value to keep
            - 'trunc_cut': Truncation threshold for singular values

        Returns
        -------
        mpo : MPOCompress
            The compressed MPO in a new MPOCompress object
        trunc_err : TruncationError
            Object containing information about the truncation error during compression

        Notes
        -----
        The compression procedure maintains the operator structure while reducing
        the bond dimension according to the specified truncation parameters.
        The method preserves the canonical form of the MPO.
        """
        # bring into right canonical_form
        mpo = self.mirror()
        mpo, Rs = mpo.left_canonical_form(absorb_R=False)
        mpo_norm = Rs[-1].replace_labels(['wL', 'wR'], ['wR', 'wL'])
        full_shape = mpo_norm.shape
        mpo_norm = mpo_norm[np.array([i for i in range(full_shape[0]-1, -1, -1)]), np.array([i for i in range(full_shape[1]-1, -1, -1)])]
        #_, mpo_norm = mpo_norm.sort_legcharge()
        mpo = mpo.mirror()
        #mpo.sort_legcharges()
        # Start Compression
        trunc_err = TruncationError()
        Ws = []
        R = npc.diag(np.ones(mpo.get_W(0).shape[0]), mpo.get_W(0).get_leg('wL'), labels=['wL', 'wR'], dtype=mpo.dtype)
        Rs = [R]
        IdLs = mpo.IdL.copy()
        IdRs = mpo.IdR.copy()
        for i in range(mpo.L):
            A = npc.tensordot(R, mpo.get_W(i), axes=['wR', 'wL'])
            IdL = IdLs[i]
            IdLnext = IdLs[i+1]
            IdR = IdRs[i+1]
            IdRlast = IdRs[i]
            W, R, IdLnext, IdR, trunc_err_new = mpo._block_SVD(A, IdL, IdLnext, IdRlast, IdR, trunc_params)
            trunc_err = trunc_err+trunc_err_new
            IdLs[i+1] = IdLnext
            IdRs[i+1] = IdR
            Rs.append(R.copy())
            Ws.append(W.copy())
        #print(Ws[0], mpo_norm)
        Ws[-1] = npc.tensordot(Ws[-1],R, axes=['wR', 'wL']).transpose(['wL', 'wR', 'p', 'p*'])
        Ws[0] = npc.tensordot(mpo_norm, Ws[0], axes=['wR', 'wL']).transpose(['wL', 'wR', 'p', 'p*'])
        IdRs[-1] = mpo.IdR[-1]
        IdLs[-1]= mpo.IdL[-1]
        mpo = MPOCompress(mpo.sites, Ws, mpo.bc, IdLs, IdRs, mpo.max_range, mpo.explicit_plus_hc)
        #perms = mpo.sort_legcharges(return_perms=True)
        #Rs = [R[perms[i], perms[i]] for i,R in enumerate(Rs)]
        return mpo, trunc_err

    def compress(self, trunc_params=None):
        """
            Compress an MPO using SVD decomposition as described in arXiv:1909.06341
            
            Args:
                trunc_params
            Returns:
                trunc_mpo, trunc_err
        """
        assert self.bc == 'finite'
        trunc_params = asConfig(trunc_params, 'trunc_params')
        with warnings.catch_warnings(record=True) as w:
            res =  self._compress_finite(trunc_params)
            for warning in w:
                if issubclass(warning.category, ComplexWarning):
                    raise RuntimeError("Raised a ComplexWarning, this might be caused by a bug in npc.Array.permute, apply monkey patch from tenpy_extensions.linalg.npc_conserved.apply_permute_patch")
        return res

    def _block_QR(self, A, IdL, IdLnext, IdRlast, IdR):
        """Performs a block QR decomposition on a matrix A with special handling of identity elements.
        This function implements a modified QR decomposition that preserves certain structure in the
        matrix while performing the decomposition A = QR. It handles special identity elements and
        maintains charge conservation in the context of tensor networks.
        Parameters
        ----------
        A : npc.Array
            Input tensor to decompose with legs (wL, wR, p, p*)
        IdL : int
            Index for left identity element
        IdLnext : int
            Next index for left identity element
        IdRlast : int
            Last index for right identity element
        IdR : int
            Index for right identity element
        Returns
        -------
        W : npc.Array
            Q matrix from QR decomposition with legs (wL, wR, p, p*)
        R : npc.Array
            R matrix from QR decomposition with legs (wL, wR)
        IdLnext_new : int
            Updated next left identity index
        IdR : int
            Updated right identity index
        Notes
        -----
        The function:
        1. Masks out identity elements
        2. Performs QR decomposition on the reduced matrix
        3. Normalizes the decomposition using the diagonal element R_oo
        4. Reconstructs full Q and R matrices preserving special structure
        5. Handles charge conservation through proper leg manipulation
        """
        mask_IdR_row = np.array([i != IdRlast for i in range(A.shape[0])])
        mask_IdR_col = np.array([i != IdR for i in range(A.shape[1])])
        old_wR = A.get_leg('wR')
        old_wL = A.get_leg('wL')
        legs_IdR = {'wL':A[IdRlast:IdRlast+1, IdR:IdR+1,:,:].get_leg('wR'), 'wR':A[IdRlast:IdRlast+1, IdR:IdR+1,:,:].get_leg('wR')}
        trivial_charge = A[IdL:IdL+1, IdLnext:IdLnext+1,:,:].get_leg('wR').get_charge(0)
        A.itranspose(['wL', 'wR', 'p', 'p*'])
        A_small = A[mask_IdR_row, mask_IdR_col, :, :]
        Q, R_small = npc.qr(A_small.combine_legs(['wL', 'p', 'p*']), inner_labels=['wR', 'wL'], inner_qconj=A.legs[0].qconj)
        Q = Q.split_legs().transpose(['wL', 'wR', 'p', 'p*'])
        IdLnext_new = Q.get_leg('wR').slices[Q.get_leg('wR').get_qindex_of_charges(trivial_charge)]
        R_oo = R_small[IdLnext_new,IdLnext]
        R_small /= R_oo
        Q *= R_oo
        
        new_legs_Q = [old_wL, Q.get_leg('wR').extend(legs_IdR['wR']), A.get_leg('p'), A.get_leg('p*')]
        new_legs_R = [new_legs_Q[1].conj(), old_wR]

        W = npc.zeros(new_legs_Q, labels=['wL', 'wR', 'p', 'p*'], dtype=self.dtype)
        W[mask_IdR_row, :-1, :, :] = Q
        W[mask_IdR_row, -1, :, :] = A[mask_IdR_row, IdR, :, :]
        W[IdRlast,-1,:,:] = A[IdRlast, IdR,:,:]
        
        R = npc.zeros(new_legs_R, labels=['wL', 'wR'], dtype=self.dtype)
        R[:-1, mask_IdR_col] = R_small
        R[-1,IdR] = 1
        IdR = R.shape[0]-1
        return W, R, IdLnext_new, IdR
    


    def _block_SVD(self, A, IdL, IdLnext, IdRlast, IdR, trunc_params):
        """
        Perform a block SVD decomposition on a tensor, handling special cases for identity matrices.
        This method performs a singular value decomposition (SVD) on a tensor while preserving
        certain identity matrix structures at specified indices. It first performs a QR
        decomposition and then SVD on the resulting R matrix, excluding identity elements.
        Parameters
        ----------
        A : npc.Array
            Input tensor to decompose
        IdL : int
            Left identity matrix index
        IdLnext : int
            Next left identity matrix index
        IdRlast : int
            Last right identity matrix index  
        IdR : int
            Right identity matrix index
        trunc_params : dict
            SVD truncation parameters
        Returns
        -------
        Q_cut : npc.Array
            Q matrix after cutting and rearranging
        R_cut : npc.Array
            R matrix after cutting and rearranging
        IdL_new : int
            New left identity index
        IdR_new : int 
            New right identity index
        trunc_err : TruncationError
            Truncation error from SVD
        Notes
        -----
        - Handles special cases where the matrix to decompose is empty or 1x1
        - Preserves identity matrix structures at specified indices
        - Uses SVD with truncation for dimensional reduction
        """
        Q, R, IdLnext_new, IdR_new = self._block_QR(A, IdL, IdLnext, IdRlast, IdR)
        mask_IdL_row = np.array([i != IdLnext_new for i in range(R.shape[0])])
        mask_IdL_col = np.array([i != IdLnext for i in range(R.shape[1])])
        mask_IdR_row = np.array([i != IdR_new for i in range(R.shape[0])])
        mask_IdR_col = np.array([i != IdR for i in range(R.shape[1])])
        # legs that that will be sliced out for svd and then later back appended
        leg_IdL_wL = R[IdLnext_new:IdLnext_new+1, IdLnext:IdLnext+1].get_leg('wL')
        leg_IdL_wR = R[IdLnext_new:IdLnext_new+1, IdLnext:IdLnext+1].get_leg('wR')
        leg_IdR_wL = R[IdR_new:IdR_new+1, IdR:IdR+1].get_leg('wL')
        leg_IdR_wR = R[IdR_new:IdR_new+1, IdR:IdR+1].get_leg('wR')
        #print(R, mask_IdL_col)
        M_small =  R[np.logical_and(mask_IdL_row,mask_IdR_row), np.logical_and(mask_IdL_col,mask_IdR_col)]
        #print(M_small)
        
        if np.prod(M_small.shape) == 0:
            legs_U = {'wL':R.get_leg('wL'),
                      'wR':leg_IdL_wR.extend(leg_IdR_wR)}
            legs_Vdag = {'wL':leg_IdL_wL.extend(leg_IdR_wL),
                         'wR':R.get_leg('wR'),
                      }
            
            IdR_new =  legs_Vdag['wL'].ind_len-1
            IdL_new = 0


            U = npc.zeros([legs_U['wL'], legs_U['wR']], dtype=self.dtype, labels = ['wL', 'wR'])
            U[np.logical_not(mask_IdL_row),IdL_new] = [1]
            U[np.logical_not(mask_IdR_row),IdR_new] = [1]

            Vdag = npc.zeros([legs_Vdag['wL'], legs_Vdag['wR']], dtype=self.dtype, labels = ['wL', 'wR'])
            Vdag[IdL_new, np.logical_not(mask_IdL_col)] = [1]
            Vdag[IdR_new, np.logical_not(mask_IdR_col)] = [1] 

            S = np.ones(legs_Vdag['wL'].ind_len)
            trunc_err = TruncationError()
        else:
            if np.prod(M_small.shape) == 1:
                U_small = M_small/M_small[0,0]
                Vdag_small = M_small/M_small[0,0]
                S_small = np.real_if_close(M_small[0,0])
                trunc_err = TruncationError()
                #print(U_small, S_small, Vdag_small)
            else:
                U_small, S_small, Vdag_small , trunc_err, renorm = svd_theta(M_small, trunc_par=trunc_params, inner_labels=['wR', 'wL'])
                M_small = Vdag_small.scale_axis(S_small, 'wL')
                S_small*= renorm
            legs_U = {'wL':R.get_leg('wL'),
                      'wR':leg_IdL_wR.extend(U_small.get_leg('wR')).extend(leg_IdR_wR)}
            legs_Vdag = {'wL':leg_IdL_wL.extend(Vdag_small.get_leg('wL')).extend(leg_IdR_wL),
                      'wR':R.get_leg('wR')}
            

            IdR_new =  legs_Vdag['wL'].ind_len-1
            IdL_new = 0

            U = npc.zeros([legs_U['wL'], legs_U['wR']], dtype= self.dtype, labels=['wL', 'wR'])
            U[np.logical_and(mask_IdL_row,mask_IdR_row), 1:-1] = U_small

            U[np.logical_not(mask_IdL_row),IdL_new] = [1]
            U[np.logical_not(mask_IdR_row),IdR_new] = [1]

            Vdag = npc.zeros([legs_Vdag['wL'], legs_Vdag['wR']], dtype= self.dtype, labels=['wL', 'wR'])
            t = R[np.logical_not(mask_IdL_row), np.logical_and(mask_IdL_col,mask_IdR_col)]
            
            Vdag[IdL_new, np.logical_not(mask_IdL_col)] = [1]
            Vdag[0:1, np.logical_and(mask_IdL_col, mask_IdR_col)] = t
            Vdag[1:-1, np.logical_and(mask_IdL_col,mask_IdR_col)] = Vdag_small
            Vdag[IdR_new, np.logical_not(mask_IdR_col)] = [1]

            S = np.ones(legs_Vdag['wL'].ind_len)
            S[1:-1] = S_small
        #print(U, Vdag)
        Q_cut = npc.tensordot(Q, U, axes = ['wR', 'wL']).transpose(['wL', 'wR', 'p', 'p*'])
        R_cut = Vdag.scale_axis(S, axis='wL').transpose(['wL', 'wR'])
        #print(Q_cut, R_cut)
        return Q_cut, R_cut, IdL_new, IdR_new, trunc_err
    


    def __add__(self, other):
        return MPOCompress.from_MPO(super().__add__(other))

