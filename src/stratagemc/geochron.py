import numpy as np
from functools import partial
import tqdm

import matplotlib.pyplot as plt


class Geochron:
    """Class for handling geochronologic constraints in stratigraphic sections."""

    def __init__(self, h, rv, dt, prob_threshold=1e-16):
        """
        Initializes the Geochron class.
        Args:
            h (array-like): List of stratigraphic heights at which temporal constraints are measured.
                The coordinate increases up section.
            rv (array-like): List of random variables representing the temporal constraints at each height. The members of this list must be objects similar to stats.dist objects with the following methods: pdf, ppf (inverse of cdf).
            dt (float): Time spacing for computing the time increment pdfs. Units are same as rvs
        Attributes:
            h (list): Sorted list of stratigraphic heights.
            rv (list): Sorted list of random variables representing the temporal constraints.
            _n_constraints (int): Number of constraints.
            _n_pairs (int): Number of pairs of constraints.
            _t_max (float): Maximum time to resolve
            _t_min (float): Minimum time to resolve
            _dt (float): Time spacing for computing the time increment pdfs.
            _t (array-like): Time grid for computing the time increment pdfs.
            lower_idx (list): List of indices into h and rv for the lower constraint in each pair.
            upper_idx (list): List of indices into h and rv for the upper constraint in each pair.
        """

        # read in heights h, and random variables rv corresponding to the geochronologic constraints
        self.h = h
        self.rv = rv
        self.dt = dt

        self._n_constraints = len(self.h)

        # sort the heights and random variable representations
        sort_idx = np.argsort(self.h)
        self.h = [self.h[idx] for idx in sort_idx]
        self.rv = [self.rv[idx] for idx in sort_idx]

        # establish unique pairs of constraints, distinguising in a consistent way the relative stratigraphic height of each
        self._pair_constraints()

        # determine the temporal range over which to grid based on probability threshold
        t_mins = [x.ppf(prob_threshold) for x in self.rv]
        t_maxs = [x.ppf(1-prob_threshold) for x in self.rv]
        self._t_max = np.max(t_maxs)
        self._t_min = np.min(t_mins)
        self._dt = dt  # temporal resolution

        # need to double sampling on grid to achieve desired temporal resolution (some form of Nyquist), hence the dt/2
        self._t = np.arange(self._t_min, self._t_max, self._dt/2)

        # verify that this grid resolves the probability distributions that were given
        for ii, _ in enumerate(self.rv):
            # first test is to ensure that 90% of probability is captured by the pdf over t1
            assert np.sum(self.rv[ii].pdf(self._t))*(self._dt/2) > 0.9, \
                f'{ii}th rv not resolvable at specified temporal sampling'

        # save probability threshold for filtering of time grid based on each constraint's pdf
        self.prob_threshold = prob_threshold

        # create time increment pdfs
        self._time_increment_pdfs()

    def _pair_constraints(self):
        """
        Determines unique pairs of constraints, ordered such that the first response is lower stratigraphically and the second is higher. outputs indices for these pairs as upper_idx and lower_idx (which index in to h, rv)
        Again, ensure that heights are increasing up section (i.e. with time)
        """
        upper_idx = []
        lower_idx = []
        for ii in range(len(self.h)-1):
            for jj in range(ii+1, len(self.h)):
                # if constraints share height, skip
                if self.h[ii] == self.h[jj]:
                    continue
                upper_idx.append(jj)
                lower_idx.append(ii)
        self.lower_idx = lower_idx
        self.upper_idx = upper_idx
        self._n_pairs = len(self.lower_idx)

    def model_weights(self, unit_heights):
        """
        Calculate the weights for units and contacts based on their bounding by
        geochron constraints.
        Args:
            unit_heights: 2d array-like (nx2) of bottom and top heights for each of n units in the section
        Returns:
            Tuple[List[List[float]], List[List[float]]]:
                - alpha: 2D list (NxL) of weights for each of N units and L pairs of geochron constraints.
                - beta: 2D list (MxL) of weights for each of M contacts and L pairs of geochron constraints.
        Raises:
            AssertionError: If any rows in alpha or beta are entirely zero, indicating unconstrained model parameters.
        Notes:
            - Run after trim_units.
        """
        unit_thicks = np.diff(unit_heights, axis=1).flatten()
        contacts = unit_heights[0:-1, 1]
        n_units = unit_heights.shape[0]
        n_contacts = n_units - 1
        # initialize
        alpha = np.zeros((n_units, self._n_pairs))
        beta = np.zeros((n_contacts, self._n_pairs))

        # loop over pairs of constraints (see notes  from 12/16/2020 for motivation of this code)
        for ii in range(self._n_pairs):
            pair_top = self.h[self.upper_idx[ii]]
            pair_bottom = self.h[self.lower_idx[ii]]
            # units
            for jj in range(n_units):
                unit_top = unit_heights[jj, 1]
                unit_bottom = unit_heights[jj, 0]
                # these factors are defined in notes; need to scan
                top_fact = np.min([np.max([(unit_top-pair_top)/unit_thicks[jj], 0]), 1])
                bot_fact = np.min(
                    [np.max([(pair_bottom-unit_bottom)/unit_thicks[jj], 0]), 1])
                frac = 1 - top_fact - bot_fact
                alpha[jj, ii] = frac

        # finally scale alpha by the thickness of the unit
        alpha = alpha * np.tile(unit_thicks, (self._n_pairs, 1)).T

        # contacts
        for ii in range(n_contacts):
            # partial bounding of contacts is not a concern, so we can just deal with one set of straddling indices
            tot_bnd_idx = self.straddling(contacts[ii])
            beta[ii, tot_bnd_idx] = 1

        # if any rows in either of matrices above are entirely zero, it means that there are model parameters that are completely unconstrained, which cannot be the case (at this point)
        assert ~np.any(np.all(alpha == 0, axis=1)), \
            'some units are not bounded at all by geochronologic constraints'
        assert ~np.any(np.all(beta == 0, axis=1)), \
            'some contacts are not bounded at all by geochronologic constraints'

        return alpha, beta

    def _time_increment_pdfs(self):
        """
        Given the provided geochronologic constraints, numerically evaluate the time increment pdfs for all pairs of constraints. These increments respect stratigrahic superosition of rv2 over rv1, meaning that it is always positive.
        """
        # get temporal bounds for each constraint that constrain given thresholds of probability (so we ignore times of zero probability)
        rv_t = []   # list of time grids for each constraint
        rv_pdf = []  # list of pdfs for each constraint
        for ii in range(self._n_constraints):
            # indices in self.t1 that contain the probability mass for the given constraint
            cur_up_bnd = self.rv[ii].ppf(1-self.prob_threshold)
            cur_low_bnd = self.rv[ii].ppf(self.prob_threshold)
            cur_idx = (self._t > cur_low_bnd) & (self._t < cur_up_bnd)
            rv_t.append(self._t[cur_idx])
            rv_pdf.append(self.rv[ii].pdf(self._t[cur_idx]))

        DTs = []
        ys = []
        for ii in tqdm.tqdm(range(self._n_pairs),
                            desc='Constructing time increment pdfs...'):
            DT, y = self._pairwise_increment_pdf(rv_t[self.lower_idx[ii]],
                                                 rv_pdf[self.lower_idx[ii]],
                                                 rv_t[self.upper_idx[ii]],
                                                 rv_pdf[self.upper_idx[ii]])
            DTs.append(DT)
            ys.append(y)

        self.DT = DTs
        self.y = ys

    def _pairwise_increment_pdf(self, t1, p1, t2, p2):
        """For a pair of constraints, numerically evaluate the time increment pdf.

        Args:
            t1 (arraylike): time grid for the lower constraint
            p1 (arraylike): probability density function for the lower constraint
            t2 (arraylike): time grid for the upper constraint
            p2 (arraylike): probability density function for the upper constraint

        Returns:
            DT (arraylike): time increment pdf for the pair of constraints
            y (arraylike): time increments over which DT is evaluated
        """
        # bounds on lower
        l1, u1 = np.min(t1), np.max(t1)
        # bounds on upper
        l2, u2 = np.min(t2), np.max(t2)

        # vector of time increments for the pair of constraints
        y = np.arange(np.max([l2-u1, 0]), u2-l1, self._dt)

        # make time grid for current pair of constraints
        T1, T2 = np.meshgrid(t1, t2)
        # make pdf grid for current pair of constraints
        P1, P2 = np.meshgrid(p1, p2)

        # for the ith pair, compute joint pdf (assuming independence)
        PJ = P1 * P2

        # now just integrate over all dt's
        # DT = np.zeros(len(y))
        # for ii, cur_y in enumerate(y):
        #     # impose stratigraphic superposition while computing time increment pdf (first term)
        #     idx = (T2 > T1) & (T2 < T1+cur_y)
        #     DT[ii] = np.sum(PJ[idx])
        idx = np.tile(T2 > T1, (len(y), 1, 1)) & \
            (np.tile(T2, (len(y), 1, 1)) < (np.tile(T1, (len(y), 1, 1)) +
                                            np.reshape(y, (-1, 1, 1))))
        # evaluate cumulative increment function here
        CDT = np.sum(PJ*idx, axis=(1, 2))

        # normalize correctly (CDF should sum to one, the max)
        CDT = CDT/np.max(CDT)
        # make into pdf
        DT = np.diff(CDT)/self._dt
    #     cur_y = cur_y[0:-1]+dt/2 # centered difference
        y = y[0:-1]  # left difference

        return DT, y

    def _pairwise_difference_rv(self, rv_t, rv_pdf, ii):
        """
        deprecated
        function to do pariwise computation of difference rv's for _rv_diff, computes the ii'th pair's difference
        """
        # bounds on lower
        l1 = rv_t[self.lower_idx[ii]][0]
        u1 = rv_t[self.lower_idx[ii]][-1]
        # bounds on upper
        l2 = rv_t[self.upper_idx[ii]][0]
        u2 = rv_t[self.upper_idx[ii]][-1]

        # current vector of time increments
        cur_y = np.arange(np.max([l2-u1, 0]), u2-l1, self._dt)

        # make time grid for current pair of constraints
        T1, T2 = np.meshgrid(rv_t[self.lower_idx[ii]], rv_t[self.upper_idx[ii]])
        # make pdf grid for current pair of constraints
        lower_pdf, upper_pdf = np.meshgrid(
            rv_pdf[self.lower_idx[ii]], rv_pdf[self.upper_idx[ii]])

        # for the ith pair, compute joint pdf (assuming independence)
        cur_joint = lower_pdf * upper_pdf

        # now just integrate over all dt's
        cur_DT = np.zeros(len(cur_y))
        for jj, _ in enumerate(cur_y):
            # impose stratigraphic superposition while computing time increment pdf (first term)
            idx = (T2 > T1) & (T2 < T1+cur_y[jj])
            cur_DT[jj] = np.sum(cur_joint[idx])

        # normalize correctly
        cur_DT = cur_DT/np.max(cur_DT)
        # make into pdf
        cur_DT = np.diff(cur_DT)/self._dt
    #     cur_y = cur_y[0:-1]+dt/2 # centered difference
        cur_y = cur_y[0:-1]  # left difference
        return cur_DT, cur_y

    def _rv_diff(self):
        """
        Deprecated
        compute difference of rv's for given pair of constraints (done once for given set of constraints)

        stores a list of random variables as pdfs capturing the probability distribution of the difference of the random variables.

        y: the dt coordinate over which each DT is evaluated
        DT: the time increment pdf for the given constraint pair
        """

        # get temporal bounds for each constraint that constrain given thresholds of probability (so we ignore times of zero probability)
        rv_t = []
        rv_pdf = []
        for ii in tqdm.tqdm(range(self._n_constraints), desc='Constructing time increment cdfs...'):
            # indices in self.t1 that contain the probability mass for the given constraint
            cur_up_bnd = self.rv[ii].ppf(1-self.prob_threshold)
            cur_low_bnd = self.rv[ii].ppf(self.prob_threshold)
            cur_idx = (self._t > cur_low_bnd) & (self._t < cur_up_bnd)
            rv_t.append(self._t[cur_idx])
            rv_pdf.append(self.rv[ii].pdf(self._t[cur_idx]))

        # make partial function compatible with pool map
        partial_pairwise_difference_rv = partial(self._pairwise_difference_rv,
                                                 self.lower_idx, self.upper_idx,
                                                 rv_t, rv_pdf, self._dt)

        # parallel case
#         pool = Pool()
#         out = list(tqdm.tqdm(pool.imap(partial_pairwise_difference_rv, range(self._n_pairs), chunksize=5), total=self._n_pairs, desc='Constructing time increment cdfs...'))
# #         DT = pool.map(partial_pairwise_difference_rv, range(self._n_pairs))
#         pool.close()

        # serial case
        out = []
        for ii in tqdm.tqdm(range(self._n_pairs), desc='Constructing time increment cdfs...'):
            out.append(partial_pairwise_difference_rv(ii))

        DT = [x[0] for x in out]
        y = [x[1] for x in out]

        self.DT = DT
        self.y = y

    def straddling(self, height):
        """
        Returns the indices of pairs of constraints that straddle a given height.
        Args:
            height (float): The height in section.
        Returns:
            array-like: The indices (into self.lower_idx, self.upper_idx) that straddle the given height.
        """

        idx = []
        for ii in range(self._n_pairs):
            if self.h[self.lower_idx[ii]] < height < self.h[self.upper_idx[ii]]:
                idx.append(ii)
        idx = np.asarray(idx)

        return idx

    def plot_constraints(self, ax=None, tol=3, scale=1):
        """
        Plot the geochronologic constraints.
        Args:
            ax (matplotlib.axes.Axes): The axes on which to plot the constraints. If None, a new figure is created.
        Returns:
            matplotlib.axes.Axes: The axes on which the constraints are plotted.
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel('Time')
            ax.set_ylabel('Height')

        for ii in range(self._n_constraints):
            cur_pdf = self.rv[ii].pdf(self._t)
            # only plot the part of the pdf that matters
            idx_pdf = np.log10(cur_pdf) > (np.max(np.log10(cur_pdf)) - tol)

            # scale it to be visible
            pdf_scale = scale/(np.max(cur_pdf[idx_pdf]) - np.min(cur_pdf[idx_pdf]))
            cur_pdf = self.h[ii] + pdf_scale * cur_pdf

            cur_h = ax.fill_between(self._t[idx_pdf],
                                    cur_pdf[idx_pdf],
                                    self.h[ii],
                                    linewidth=1,
                                    alpha=0.7,
                                    color='lightgrey')

        return ax