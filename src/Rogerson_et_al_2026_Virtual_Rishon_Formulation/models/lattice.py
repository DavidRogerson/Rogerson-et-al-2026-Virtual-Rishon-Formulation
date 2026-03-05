import numpy as np
from tenpy.models.lattice import Lattice, IrregularLattice
from copy import deepcopy, copy
from tenpy.networks.site import set_common_charges
from tenpy.tools.misc import to_iterable

#2025-10-12 Implemented a more robust gauge order called: before_target_new
#   Works like before_target but for all lattices, Problem it is incompatible with old implementation on periodic rings, (flipps last two gauge terms)
#   Because of this I need to keep the old one, such that previous results are compatible, what a bummer...
#   for details see notebooks/playground/25-10-10_BetterGaugeOrder.ipynb
class LatticeGaugeTheoryLattice(IrregularLattice):
    """A variant of :class:`tenpy.models.lattice.Lattice` that replaces the lattice sites with a set of matter sites, and adds one set of gauge sites per nearest neighbor pair.

    The gauge sites are added to the unit cell where the source of the edge is located.
    Gauge symmetry is locally conserved by the virtual rishon construction.
    Includes a macro unit cell, which is the full tiling of the lattice unit cell, allowing each entry to have its own charge info. This enables extensive charge conservation.

    Bug:
    ----
    Currently, open boundary conditions with gauge_order = 'before_target' and remove_missing_bonds = True do not work correctly.
    I assume that checking for the reverse connection, does add the bond always. I will have to fix this later.

    Todo:
    ----
    - Add convinience access to matter_site_A gauge site matter_site_B, similar to `self.simple_lattice.pairs['nearest_neighbors']` for the simple lattice.
    - Add convinience access to unit_cell indices based on site_type.
    - Add convinience access to gauge start operator on every node.
    - Add convinience access to magnetic field operator plaquets. (Not super trivial).

    Parameters
    ----------
    simple_lattice : Lattice
        The base lattice to which the multispecies matter and gauge sites are added.
    matter_sites : list of Site
        The matter sites to be added to the lattice.
    gauge_sites : list of Site
        The gauge sites to be added to the lattice, for each nearest neighbor pair.
    VirtualRishonConstructor : callable
        Function to construct virtual rishon sites from source and target gauge sites.
    matter_names : list of str
        The names of the matter sites.
    gauge_names : list of str
        The names of the gauge sites.
    remove_missing_bonds : bool, optional
        Whether to remove gauge sites not connected to any matter site (default: True).
    new_charges : str, optional
        How to set new charges for the sites (default: 'same').
    """
    def __init__(self, simple_lattice, matter_sites, gauge_sites, VirtualRishonConstructor, matter_names, gauge_names,
                 remove_missing_bonds=True, new_charges='same', charge_mod = None, gauge_order=None):

        simple_Lu = simple_lattice.Lu
        if simple_Lu is None:
            simple_Lu = len(simple_lattice.unit_cell)
        N_matter_species = len(matter_sites)
        N_gauge_species = len(gauge_sites)
        N_species = N_matter_species + N_gauge_species

        simple_lnn = len(simple_lattice.pairs['nearest_neighbors'])
        unit_cell = sum([deepcopy(list(matter_sites)) for _ in range(simple_Lu)] + [deepcopy(list(gauge_sites)) for _ in range(simple_lnn)], start = [])

        if matter_names is None:
                    matter_names = [str(i) for i in range(N_matter_species)]
        if len(matter_names) != N_matter_species:
            raise ValueError("need exactly one name for each species,"
                         f"but got {matter_names!r} for {matter_names!r}")
        
        if gauge_names is None:
            gauge_names = [f'gauge_{i}' for i in range(N_gauge_species)]
        if len(gauge_names) != N_gauge_species:
            raise ValueError("need exactly one name for each species,"
                         f"but got {gauge_names!r} for {gauge_names!r}")
        
        self.simple_lattice = simple_lattice
        unit_cell_positions =  np.array(sum([self.points_on_circle(p, r=0.025, N=N_matter_species) for p in self.simple_lattice.unit_cell_positions], start=[])
                                             + sum([self.points_on_circle(self.calc_bond_positions(u1,u2,dx), r=0.025, N=N_gauge_species) for u1, u2, dx in self.simple_lattice.pairs['nearest_neighbors']], start = [])
        )
        if charge_mod is None:
            charge_mod = simple_lattice.Ls
        else:
            assert np.shape(charge_mod) == np.shape(simple_lattice.Ls), "charge_mod must have the same shape as simple_lattice.Ls"
        self.VirtualRishonConstructor = VirtualRishonConstructor
        self.remove_missing_bonds = remove_missing_bonds
        self.new_charges = new_charges
        self.N_species = N_species
        self.N_matter_species = N_matter_species
        self.N_gauge_species = N_gauge_species
        self.matter_names = matter_names
        self.gauge_names = gauge_names
        self.species_names = matter_names + gauge_names
        self.simple_Lu = simple_Lu
        self.remove = []  # stores the sites that are not part of the order, i.e. the gauge sites that are not connected to any matter site
        new_pairs = self._generate_new_pairs()
        if gauge_order is None:
            gauge_order = 'after_source'
        self.gauge_order = gauge_order


        Lattice.__init__(
            self,
            simple_lattice.Ls,
            unit_cell,
            bc=simple_lattice.boundary_conditions,
            bc_MPS=simple_lattice.bc_MPS,
            basis=simple_lattice.basis,
            positions=unit_cell_positions,
            pairs=new_pairs)

        self.order = self._simple_order_to_self_order(simple_lattice.order)
        self.macro_cell = self.generate_macro_cell(matter_sites, gauge_sites, charge_mod)
        unit_cell = [None for _ in range(simple_Lu * N_matter_species + simple_lnn * N_gauge_species)]
        for i,order_i in enumerate(self.order):
            if not (None in unit_cell):
                break
            else:
                u = order_i[-1]
                if unit_cell[u] is None:
                    site = self.macro_cell[i]
                    unit_cell[u] = site
        self.unit_cell = unit_cell



    @Lattice.order.setter
    def order(self, order_):
        # very similar to HelicalLattice.order setter
        self._order = np.array(order_, dtype=np.intp)
        _mps_vals_idx = np.full(np.max(self._order, axis=0)+1,self._REMOVED, dtype=np.intp)
        for i, o in enumerate(order_):
            _mps_vals_idx[tuple(o)] = i
        self._mps2lat_vals_idx = _mps_vals_idx

        self._mps2lat_vals_idx_fix_u = [_mps_vals_idx[..., i] for i in range(_mps_vals_idx.shape[-1])]
        # this defines `self._perm`
        perm = np.full([np.prod(self.shape)], self._REMOVED)
        perm[np.sum(self._order * self._strides, axis=1)] = np.arange(len(order_))
        self._perm = perm
        # use advanced numpy indexing...
        #self._mps2lat_vals_idx = np.empty(self.shape, np.intp)
        #self._mps2lat_vals_idx[tuple(order_.T)] = np.arange(self.N_sites)
        # versions for fixed u
        self._mps_fix_u = []
        #self._mps2lat_vals_idx_fix_u = []
        for u in range(len(self.unit_cell)):
            mps_fix_u = np.nonzero(order_[:, -1] == u)[0]
            self._mps_fix_u.append(mps_fix_u)
            #bugfix missing in setter function ?
            # mps2lat_vals_idx = np.empty(self.Ls, np.intp)
            # mps2lat_vals_idx[tuple(order_[mps_fix_u, :-1].T)] = np.arange(self.N_cells)
            # self._mps2lat_vals_idx_fix_u.append(mps2lat_vals_idx)
            # end bugfix
        self._mps_fix_u = tuple(self._mps_fix_u) 
        self.N_sites = len(order_)
        _, counts = np.unique(order_[:, 0], return_counts=True)
        if np.all(counts == counts[0]):
            self.N_sites_per_ring = counts[0]
        else:
            self.N_sites_per_ring = max(counts)
        
    def _generate_new_pairs(self):
        N_sp = self.N_species
        names = self.species_names
        new_pairs = {}
        # note: inline species_u_to_simple_u and simple_u_to_species methods here
        errmsg = "duplicate key %s for pairs; use different species names!"
        for pair_key, pair_val in self.simple_lattice.pairs.items():
            pair_val_all = []
            pair_val_diag = []
            for sp_idx1, sp_name1 in enumerate(names):
                for sp_idx2, sp_name2 in enumerate(names):
                    pair_key_sp = f"{pair_key}_{sp_name1}-{sp_name2}"
                    if pair_key_sp in new_pairs:
                        raise ValueError(errmsg % pair_key_sp)
                    pair_val_sp = []
                    for (u1, u2, dx) in pair_val:
                        pair_val_sp.append((u1 * N_sp + sp_idx1, u2 * N_sp + sp_idx2, dx))
                    new_pairs[pair_key_sp] = pair_val_sp
                    pair_val_all.extend(pair_val_sp)
                    if sp_idx1 == sp_idx2:
                        pair_val_diag.extend(pair_val_sp)
            for key_sum, pair_val_sum in [('all-all', pair_val_all), ('diag', pair_val_diag)]:
                pair_key_sum = f"{pair_key}_{key_sum}"
                if pair_key_sum in new_pairs:
                    if pair_key_sp in new_pairs:
                        raise ValueError(errmsg % pair_key_sum)
                new_pairs[pair_key_sum] = pair_val_sum
        dx = [0] * self.simple_lattice.dim
        for sp_idx1, sp_name1 in enumerate(names):
            for sp_idx2, sp_name2 in enumerate(names):
                if sp_idx2 <= sp_idx1:
                    continue # fully onsite!
                onsite_pair_key = f"onsite_{sp_name1}-{sp_name2}"
                onsite_pair_val = [(u * N_sp + sp_idx1, u * N_sp + sp_idx2, dx)
                                   for u in range(self.simple_Lu)]
                if onsite_pair_key in new_pairs:
                    raise ValueError(errmsg % onsite_pair_key)
                new_pairs[onsite_pair_key] = onsite_pair_val
        return new_pairs


    def mps2lat_values(self, A, axes=0, u=None):
        """Reshape/reorder `A` to replace an MPS index by lattice indices.

        Parameters
        ----------
        A : ndarray
            Some values. Must have ``A.shape[axes] = self.N_sites`` if `u` is ``None``, or
            ``A.shape[axes] = self.N_cells`` if `u` is an int.
        axes : (iterable of) int
            chooses the axis which should be replaced.
        u : ``None`` | int
            Optionally choose a subset of MPS indices present in the axes of `A`, namely the
            indices corresponding to ``self.unit_cell[u]``, as returned by :meth:`mps_idx_fix_u`.
            The resulting array will not have the additional dimension(s) of `u`.

        Returns
        -------
        res_A : ndarray
            Reshaped and reordered version of A. Such that MPS indices along the specified axes
            are replaced by lattice indices, i.e., if MPS index `j` maps to lattice site
            `(x0, x1, x2)`, then ``res_A[..., x0, x1, x2, ...] = A[..., j, ...]``.

        Examples
        --------
        Say you measure expectation values of an onsite term for an MPS, which gives you an 1D array
        `A`, where `A[i]` is the expectation value of the site given by ``self.mps2lat_idx(i)``.
        Then this function gives you the expectation values ordered by the lattice:

        .. testsetup :: mps2lat_values

            lat = tenpy.models.lattice.Honeycomb(10, 3, None)
            A = np.arange(60)
            C = np.arange(60*60).reshape(60, 60)
            A_res = lat.mps2lat_values(A)

        .. doctest :: mps2lat_values

            >>> print(lat.shape, A.shape)
            (10, 3, 2) (60,)
            >>> A_res = lat.mps2lat_values(A)
            >>> A_res.shape
            (10, 3, 2)
            >>> bool(A_res[tuple(lat.mps2lat_idx(5))] == A[5])
            True

        If you have a correlation function ``C[i, j]``, it gets just slightly more complicated:

        .. doctest :: mps2lat_values

            >>> print(lat.shape, C.shape)
            (10, 3, 2) (60, 60)
            >>> lat.mps2lat_values(C, axes=[0, 1]).shape
            (10, 3, 2, 10, 3, 2)

        If the unit cell consists of different physical sites, an onsite operator might be defined
        only on one of the sites in the unit cell. Then you can use :meth:`mps_idx_fix_u` to get
        the indices of sites it is defined on, measure the operator on these sites, and use
        the argument `u` of this function.

        .. doctest :: mps2lat_values

            >>> u = 0
            >>> idx_subset = lat.mps_idx_fix_u(u)
            >>> A_u = A[idx_subset]
            >>> A_u_res = lat.mps2lat_values(A_u, u=u)
            >>> A_u_res.shape
            (10, 3)
            >>> bool(np.all(A_res[:, :, u] == A_u_res[:, :]))
            True
        """
        axes = to_iterable(axes)
        if len(axes) > 1:
            axes = [(ax + A.ndim if ax < 0 else ax) for ax in axes]
            for ax in reversed(sorted(axes)):  # need to start with largest axis!
                A = self.mps2lat_values(A, ax, u)  # recursion with single axis
            return A
        # choose the appropriate index arrays
        if u is None:
            idx = self._mps2lat_vals_idx
        else:
            idx = self._mps2lat_vals_idx_fix_u[u]
        # A = np.take(A, idx, axis=axes[0], mode='clip')
        # A[idx == self._REMOVED] = np.nan
        sel = idx != self._REMOVED  # True where you want to keep the value
        idx_clipped = np.clip(idx, 0, A.shape[axes[0]] - 1)

        A_taken = np.take(A, idx_clipped, axis=axes[0])
        # Now mask out removed entries
        mask_removed = idx == self._REMOVED

        # To mask `A_taken` along axis=axes[0]
        slicer = [slice(None)] * A.ndim
        slicer[axes[0]] = mask_removed
        A_taken[tuple(slicer)] = np.nan

        A = A_taken

        return A


    def site(self, i):
        return self.macro_cell[i]
    
    def mps_sites(self):
        return self.macro_cell
    
    def ordering(self, order):
        """Define orderings as for the `simple_lattice` with priority for within the unit cell.

        See :meth:`Lattice.ordering` for arguments.
        """
        simple_order = self.simple_lattice.ordering(order)
        return self._simple_order_to_self_order(simple_order)

    def _simple_order_to_self_order(self, simple_order):
        """
        This function loops throught the simple order and converts it to the lattice with gauge degrees.

        For each entry in the simple order it adds all the different matter species.
        Then it loops through all possible nearest neighbor couplings starting at this site, and adds the gauge degree of freedoms as the next entries in the order.
        The it moves on the next entry in the simple order.
        Parameters
        ----------
        simple_order : list of tuples
            The order of the simple lattice, where each entry is a tuple of the form (x, y, u) or (x, u).
            Here (x, y) are the coordinates of the site in the unit cell, and u is the index of the unit cell element.
        Returns
        -------
        order :list of tuples
            The order of the gauge lattice, where each entry is a tuple of the form (x, y, u) or (x, u).
            the unit cell positions (x,y) follow the simple lattice.
            u is more complicated, the first $Nf$ us are the matter fields the latter $nB * Nr$ are the rotor fields
            The rotor fields are inserted into the ordering dependent on the choosen insertion method, before target or after_source. 
        """
        #simple_order is of form x,y,u

        new_order = []
        sources = []
        targets = []
        seen_simple_lat_mps_idx = []

        for link_idx, coupling in enumerate(self.simple_lattice.pairs['nearest_neighbors']):
            s, t, _, _ = self.simple_lattice.possible_couplings(*coupling)
            sources.append(s)
            targets.append(t)
        sources, targets

        for simple_mps_i, order_i in enumerate(simple_order):
            p = order_i[:-1]  # all but the last entry are the lattice coordinates
            u = order_i[-1]   # the last entry is the unit cell index
            if self.gauge_order == 'after_source':
                for matter_idx in range(self.N_matter_species):
                    new_order.append((*p,u*self.N_matter_species + matter_idx))
                for link_idx, coupling in enumerate(self.simple_lattice.pairs['nearest_neighbors']):
                    if coupling[0] == u:
                        source = self.simple_lattice.possible_couplings(*coupling)[0]
                        check_source = np.argwhere(source == simple_mps_i)
                        if check_source.size == 1:
                            for gauge_species_idx in range(self.N_gauge_species):
                                new_order.append((*p, self.simple_u_to_species_u('gauge', link_idx, gauge_species_idx)))
            elif self.gauge_order ==  'before_target':
                #introduces new simple_lat mps index, checks if any gauge links, are between all the previous simple_lat sites, and the new sites, introduce coresponding gauge fields, then add new matter sites.
                for link_idx, coupling in enumerate(self.simple_lattice.pairs['nearest_neighbors']):
                    sources, targets, _, _ = self.simple_lattice.possible_couplings(*coupling)
                    check_source_targets = np.argwhere(np.logical_and(sources <= simple_mps_i, targets == simple_mps_i))
                    check_source_targets_reverse = np.argwhere(np.logical_and(targets <= simple_mps_i, sources == simple_mps_i))
                    check_source = np.argwhere(sources == simple_mps_i)
                    if check_source.size == 0:
                        if check_source_targets_reverse.size == 0:
                            if self.remove_missing_bonds:
                                for gauge_species_idx in range(self.N_gauge_species):
                                    self.remove.append((*p, self.simple_u_to_species_u('gauge', link_idx, gauge_species_idx)))
                            else:
                                for gauge_species_idx in range(self.N_gauge_species):
                                    new_order.append((*p, self.simple_u_to_species_u('gauge', link_idx, gauge_species_idx)))   
                    else:
                        if check_source_targets.size > 1:
                            raise ValueError(f"Found multiple sources for {simple_mps_i} in {sources} at position {p} and link index {link_idx}.")
                        elif check_source_targets.size == 1:
                            p_for_gauge_source = simple_order[check_source_targets[0, 0]][:-1]  # get the position of the source site
                            # we have a source, so we add the gauge species to the order
                            for gauge_species_idx in range(self.N_gauge_species):
                                new_order.append((*p_for_gauge_source, self.simple_u_to_species_u('gauge', link_idx, gauge_species_idx)))
                        if check_source_targets_reverse.size > 1:
                            raise ValueError(f"Found multiple targets for {simple_mps_i} in {targets} at position {p} and link index {link_idx}.")
                        elif check_source_targets_reverse.size == 1:
                            p_for_gauge_target = simple_order[check_source_targets_reverse[0, 0]][:-1]
                            # we have a target, so we add the gauge species to the order
                            for gauge_species_idx in range(self.N_gauge_species):
                                new_order.append((*p_for_gauge_target, self.simple_u_to_species_u('gauge', link_idx, gauge_species_idx)))
                for matter_idx in range(self.N_matter_species):
                    new_order.append((*p,u*self.N_matter_species + matter_idx))

            elif self.gauge_order ==  'before_target_new':
                if simple_mps_i != 0:
                    #introduces new simple_lat mps index, checks if any gauge links, are between all the previous simple_lat sites, and the new sites, introduce coresponding gauge fields, then add new matter sites.
                    for link_idx, coupling in enumerate(self.simple_lattice.pairs['nearest_neighbors']):
                        s = sources[link_idx]
                        t = targets[link_idx]
                        if simple_mps_i in s:
                            pos_s = np.argwhere(simple_mps_i==s).flatten()
                            if t[pos_s][0] in seen_simple_lat_mps_idx:
                                for gauge_species_idx in range(self.N_gauge_species):
                                    new_order.append((*p, self.simple_u_to_species_u('gauge', link_idx, gauge_species_idx)))
                        if simple_mps_i in t:
                            pos_t = np.argwhere(simple_mps_i==t).flatten()
                            if s[pos_t][0] in seen_simple_lat_mps_idx:
                                p_s = simple_order[s[pos_t][0]][:-1]
                                for gauge_species_idx in range(self.N_gauge_species):
                                    new_order.append((*p_s, self.simple_u_to_species_u('gauge', link_idx, gauge_species_idx)))
                for matter_idx in range(self.N_matter_species):
                    new_order.append((*p,u*self.N_matter_species + matter_idx))
                seen_simple_lat_mps_idx.append(simple_mps_i)
        
        Ls = self.simple_lattice.Ls
        all_sites = []
        unit_cell_indices = range(len(self.unit_cell))

        # Check the number of dimensions and generate all possible sites
        if len(Ls) == 1:
            # This is a 1D lattice
            Lx = Ls[0]
            for x in range(Lx):
                for u in unit_cell_indices:
                    # For 1D, a site might be represented by (x, u)
                    all_sites.append((x, u))
        elif len(Ls) == 2:
            # This is a 2D lattice
            Lx, Ly = Ls
            for x in range(Lx):
                for y in range(Ly):
                    for u in unit_cell_indices:
                        all_sites.append((x, y, u))
        else:
            # It's good practice to handle unexpected cases
            raise ValueError(f"Unsupported lattice dimension: {len(Ls)}")

        # Use set difference for an efficient way to find removed sites.
        # This is much faster than using list.remove() in a loop.
        all_sites_set = set(all_sites)
        existing_couplings_set = set(new_order)

        self.remove = list(all_sites_set - existing_couplings_set)

        order = np.array(new_order, dtype=int)
        return order

    def generate_macro_cell(self, matter_sites, gauge_sites, charge_mod):
        matter_sites_dict = {}
        for p in np.ndindex(self.simple_lattice.Ls):
            for u in range(self.simple_Lu):
                matter_sites_dict[(*p, u)] = [copy(site) for site in matter_sites]
        target_sites = {}
        source_sites = {}
        # create the target and source sites for the gauge sites
        for link_idx , (u1, u2, dx) in enumerate(self.simple_lattice.pairs['nearest_neighbors']):
            mps1, mps2, _, _ = self.simple_lattice.possible_couplings(u1, u2, dx)
            for i in mps1:
                # i is the index of the source site in the simple lattice
                simple_lat_index = self.simple_lattice.mps2lat_idx(i).tolist()
                p = simple_lat_index[:-1]
                u = simple_lat_index[-1]
                source_sites[(*p, u,link_idx)] = [copy(site) for site in gauge_sites]
            for j in mps2:
                simple_lat_index = self.simple_lattice.mps2lat_idx(j).tolist()
                p = simple_lat_index[:-1]
                u = simple_lat_index[-1]
                # j is the index of the target site in the simple lattice
                target_sites[(*p,u, link_idx)] = [copy(site) for site in gauge_sites]
        # make one charge per simple lattice site, with all matter sites and all target and source sites connected to it.
        for site_key, sites in matter_sites_dict.items():
            current_sites = list(sites)
            for bond_key, bond_sites in target_sites.items():
                if np.array_equal(bond_key[:-1],site_key):
                    current_sites += bond_sites
            for bond_key, bond_sites in source_sites.items():
                if np.array_equal(bond_key[:-1],site_key):
                    current_sites += bond_sites
            # set the common charge for all sites in the current_sites
            p, u = site_key[:-1], site_key[-1]
            p = tuple([pi% cm_i for pi, cm_i in zip(p, charge_mod)])  # ensure p is within the bounds of the simple lattice
            set_common_charges(current_sites, new_charges=self.new_charges, new_names=[f'G_{(*p, u)}'])
        # expand the different matter sites to new unit cell
        all_sites_temp = sum([val for key, val in matter_sites_dict.items()] + [val for key, val in target_sites.items()]+[val for key, val in source_sites.items()] , start=[])
        set_common_charges(all_sites_temp)
        all_sites_dict = {}
        for key, sites in matter_sites_dict.items():
            # key is a tuple of the form (p, u) where p is the position in the simple lattice and u is the index of the unit cell element
            # sites is a list of matter sites
            p = key[:-1]  # all but the last entry are the lattice coordinates
            u = key[-1]   # the last entry is the unit cell index
            for i, site in enumerate(sites):
                all_sites_dict[(*p, self.simple_u_to_species_u('matter', u, i))] = site    
        # merge the target and source sites and append them to the matter site now all site dict
        for link_idx , (u1, u2, dx) in enumerate(self.simple_lattice.pairs['nearest_neighbors']):
            mps1, mps2, _, _ = self.simple_lattice.possible_couplings(u1, u2, dx)
            for source_mpsi, target_mpsj in zip(mps1, mps2):
                p_source = self.simple_lattice.mps2lat_idx(source_mpsi)[:-1].tolist()
                u_source = self.simple_lattice.mps2lat_idx(source_mpsi)[-1].tolist()
                sources = source_sites[(*p_source, u_source, link_idx)]
                p_target = self.simple_lattice.mps2lat_idx(target_mpsj)[:-1].tolist()
                u_target = self.simple_lattice.mps2lat_idx(target_mpsj)[-1].tolist()
                targets = target_sites[(*p_target, u_target, link_idx)]
                for gauge_species_idx, site in enumerate(self.VirtualRishonConstructor(sources, targets)):
                    p = tuple(self.simple_lattice.mps2lat_idx(source_mpsi)[:-1].tolist())
                    u = self.simple_u_to_species_u('gauge', link_idx, gauge_species_idx)
                    all_sites_dict[(*p, u)] = site
        # make a list of all the sites based on the order of the lattice
        all_sites = [all_sites_dict[tuple(key.tolist())] for key in self.order]
        return all_sites



    def species_u_to_simple_u(self, u_idx):
        """Returns which kind of site a given unit cell index corresponds to.
        For matter sites it return 'matter', the index of the simple unit cell as well as the matter species index.
        For gauge sites it returns 'gauge', the index of the nn-coupling as well as the gauge species index.
        Parameters
        ----------
        u_idx : int
            The index of the unit cell element.
        Returns
        -------
        tuple of str, int, int
            A tuple containing the type of the site ('matter' or 'gauge'), the index of the simple unit cell element, and the species index.
            For matter sites, the species index is the index of the matter species.
            For gauge sites, the species index is the index of the gauge species.
        """
        if u_idx < self.simple_Lu * self.N_matter_species:
            # matter site
            u_simple = u_idx // self.N_matter_species
            m_idx = u_idx % self.N_matter_species
            return 'matter', u_simple, m_idx
        else:
            # gauge site
            u_gauge = (u_idx - self.simple_Lu * self.N_matter_species) // self.N_gauge_species
            g_idx = (u_idx - self.simple_Lu * self.N_matter_species) % self.N_gauge_species
            return 'gauge', u_gauge, g_idx

    def simple_u_to_species_u(self, site_type, u_simple_gauge, species_idx):
        """Converts a simple unit cell index and a species index to the corresponding unit cell element index in this lattice.
        Parameters
        ----------
        site_type : str
            The type of the site ('matter' or 'gauge').
        u_simple_gauge : int
            The index of the simple unit cell element.
        species_idx : int
            The index of the species (matter or gauge).
        Returns
        -------
        int
            The index of the unit cell element in this lattice.
        """
        if site_type == 'matter':
            return u_simple_gauge * self.N_matter_species + species_idx
        elif site_type == 'gauge':
            return self.simple_Lu * self.N_matter_species + u_simple_gauge * self.N_gauge_species + species_idx
        else:
            raise ValueError(f"Unknown type: {site_type}. Expected 'matter' or 'gauge'.")

    def calc_bond_positions(self, u1, u2, dx):
        """Calculates the midpoint of a bond defined by the two unit cell element indices, as well as the unit cell displacement vector.
        Parameters
        ----------
        u1 :int
            The index of the first unit cell element.
        u2 :int
            The index of the second unit cell element.
        dx : tuple of int
            The displacement vector from the first to the second unit cell element.
        Returns
        -------
        position : tuple of float
            The midpoint of the bond defined by the two unit cell elements and the displacement vector.
            The position is calculated as the average of the positions of the two unit cell elements, plus the displacement vector.
            """
        pos1 = self.simple_lattice.unit_cell_positions[u1]
        pos2 = self.simple_lattice.unit_cell_positions[u2]
        x = 0.5
        return x*pos1 + (1-x)*(pos2 + dx@self.simple_lattice.basis)
    

    def points_on_circle(self, center, r, N):
        """
        Generate N points evenly distributed around a center in 2D, at distance r.

        Parameters
        ----------
        center : tuple of float
            The (x, y) coordinates of the center point.
        r : float
            The radius (distance from center).
        N : int
            Number of points to generate.

        Returns
        -------
        points : list of tuple
            List of (x, y) coordinates for each point.
        """
        if N == 1:
            return [center]
        else:
            if len(center) ==1:
                dxs = np.linspace(-4*r, 4*r, N)
                points = [center+dx for dx in dxs]
            elif len(center) == 2:
                # Generate N points evenly spaced around the circle
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                cx, cy = center
                points = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
            else:
                raise ValueError("Center must be a 1D or 2D point.")
            return points
        