import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import LegCharge
from tenpy.tools.misc import inverse_permutation
from logging import info
def apply_permute_patch():
    """Monkey-patch the `permute` method of `npc.Array` to fix a bug where the data type (`dtype`) is not preserved,
    which can lead to loss of imaginary parts in complex arrays.

    This function replaces the original `permute` method of `npc.Array` with a corrected version that ensures
    new data blocks are created with the same `dtype` as the original array. The patch addresses an issue where
    the missing `dtype=self.dtype` argument in the creation of new blocks could result in incorrect data types.

    The patched `permute` method applies a permutation to the indices of a specified axis, similar to `np.take`
    with a 1D array. It returns a copy of the array with the specified axis permuted according to the given permutation.

    Usage:
        apply_permute_patch()
        # After calling this function, npc.Array.permute will use the patched version.

    Notes:
        - This patch is intended as a workaround for a specific bug in the original implementation.
        - The method is relatively slow and should only be used when necessary.
        - For general permutations that do not mix charge blocks, consider using `sort_legcharge` for better performance.

    See Also:
        npc.Array.permute
        sort_legcharge
        """
    
    info( "Applying monkey patch `tenpy_extensions.linalg.npc_conserved.apply_permute_patch` to fix bug in npc.Array.permute that can drop imaginary parts")
    def permute(self, perm, axis):
        """Apply a permutation in the indices of an axis.

        Similar as np.take with a 1D array.
        Roughly equivalent to ``res[:, ...] = self[perm, ...]`` for the corresponding `axis`.
        Note: This function is quite slow, and usually not needed!

        Parameters
        ----------
        perm : array_like 1D int
            The permutation which should be applied to the leg given by `axis`.
        axis : str | int
            A leg label or index specifying on which leg to take the permutation.

        Returns
        -------
        res : :class:`Array`
            A copy of self with leg `axis` permuted, such that
            ``res[i, ...] = self[perm[i], ...]`` for ``i`` along `axis`.

        See also
        --------
        sort_legcharge : can also be used to perform a general permutation.
            Preferable, since it is faster for permutations which don't mix charge blocks.
        """
        axis = self.get_leg_index(axis)
        perm = np.asarray(perm, dtype=np.intp)
        oldleg = self.legs[axis]
        if len(perm) != oldleg.ind_len:
            raise ValueError("permutation has wrong length")
        inv_perm = inverse_permutation(perm)
        newleg = LegCharge.from_qflat(self.chinfo, oldleg.to_qflat()[perm], oldleg.qconj)

        res = self.copy(deep=False)  # data is replaced afterwards
        res.legs[axis] = newleg
        qdata_axis = self._qdata[:, axis]
        new_block_idx = [slice(None)] * self.rank
        old_block_idx = [slice(None)] * self.rank
        data = []
        qdata = {}  # dict for fast look up: tuple(indices) -> _data index
        for old_qind, (beg, end) in enumerate(oldleg._slice_start_stop()):
            old_range = range(beg, end)
            for old_data_index in np.nonzero(qdata_axis == old_qind)[0]:
                old_block = self._data[old_data_index]
                old_qindices = self._qdata[old_data_index]
                new_qindices = old_qindices.copy()
                for i_old in old_range:
                    i_new = inv_perm[i_old]
                    qi_new, within_new = newleg.get_qindex(i_new)
                    new_qindices[axis] = qi_new
                    # look up new_qindices in `qdata`, insert them if necessary
                    new_data_ind = qdata.setdefault(tuple(new_qindices), len(data))
                    if new_data_ind == len(data):
                        # insert new block
                        data.append(np.zeros(res._get_block_shape(new_qindices), dtype=self.dtype)) # this dtype=self.dtype fixes the bug
                    new_block = data[new_data_ind]
                    # copy data
                    new_block_idx[axis] = within_new
                    old_block_idx[axis] = i_old - beg
                    new_block[tuple(new_block_idx)] = old_block[tuple(old_block_idx)]
        # data blocks copied
        res._data = data
        res._qdata_sorted = False
        res_qdata = res._qdata = np.empty((len(data), self.rank), dtype=np.intp)
        for qindices, i in qdata.items():
            res_qdata[i] = qindices
        return res
    npc.Array.permute = permute