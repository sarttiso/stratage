import logging
import sys

import numpy as np

from numba import njit

import pytensor.tensor as pt
from pytensor.graph import Apply, Op

from scipy.optimize import minimize_scalar, lsq_linear

from stratagemc.geochron import Geochron

from stratagemc import __version__

__author__ = "Adrian Tasistro-Hart"
__copyright__ = "Adrian Tasistro-Hart"
__license__ = "GPL-3.0-only"

_logger = logging.getLogger(__name__)


def geochron_height_check(unit_heights, geochron_heights):
    """
    Ensure that geochron heights are within units and not at contacts. Also ensure that geochron heights are within the section. Run before trimming section to the top and bottom of the geochron constraints.
    Args:
        unit_heights: Bottom and top heights of units in section. The heights are given as a 2d array-like (nx2) of bottom and top heights for each of n units, increasing downwards in array
        geochron_heights: Heights of geochron constraints in the section, array-like
    Raises:
        AssertionError: If any of the geochron heights are below the section or above the section.
        AssertionError: If any of the geochron heights are at contacts.
    """
    # check that geochron heights are within the section
    assert np.all(geochron_heights >= np.min(unit_heights)
                  ), 'geochron heights are below section'
    assert np.all(geochron_heights <= np.max(unit_heights)
                  ), 'geochron heights are above section'
    # check that geochron heights are not at contacts
    offending_heights = np.intersect1d(geochron_heights, np.unique(unit_heights))
    assert len(offending_heights) == 0, offending_heights


def trim_units(unit_heights, geochron_heights):
    """
    Trim the top and bottom of the section to the top and bottom of the geochron constraints. Run after geochron_height_check.
    Args:
        unit_heights: Heights of contacts in section, including top and bottom.
            2d array-like (nx2) of bottom and top heights for each of n units
        geochron_heights: Heights of geochron constraints in section
    Returns:
        unit_heights_trim: Trimmed unit heights after adjusting for the top and bottom units
    Raises:
        None
    """
    geochron_top = np.max(geochron_heights)
    geochron_bottom = np.min(geochron_heights)

    below_idx = np.any(unit_heights < geochron_top, axis=1)
    above_idx = np.any(unit_heights > geochron_bottom, axis=1)

    n_trimmed_units = np.sum(~(below_idx & above_idx))
    _logger.info(f"Trimmed {n_trimmed_units} units outside of geochron constraints.")

    unit_heights_trim = unit_heights[below_idx & above_idx]

    # adjust heights for the top and bottom units
    unit_heights_trim[-1, 1] = geochron_top
    unit_heights_trim[0, 0] = geochron_bottom

    return unit_heights_trim


def get_times(sed_rates, hiatuses, units):
    """
    Floating times array for units.
    Args:
        sed_rates: Sedimentation rates for each unit, array-like
        hiatuses: Hiatuses between units, array-like
        units: Heights of contacts in section, including top and bottom.
            2d array-like (nx2) of bottom and top heights for each of n units
    Returns:
        times: Floating times array for units, array-like
    """
    n_units = units.shape[0]
    assert len(sed_rates) == n_units, 'must have sed rate for each unit'
    assert len(hiatuses) == n_units-1, 'must have n_units-1 hiatuses'

    # unit thicknesses
    thicks = np.diff(units, axis=1).squeeze()
    # time in each unit
    unit_times = thicks/sed_rates
    times = np.zeros(units.shape)
    for ii in range(n_units-1):
        times[ii, 1] = times[ii, 0] + unit_times[ii]
        times[ii+1, 0] = times[ii, 1] + hiatuses[ii]
    times[-1, 1] = times[-1, 0] + unit_times[-1]

    return times


@njit
def sigmoid(t, scale=0.001):
    """Sigmoid function for numerical stability.

    Args:
        t (float | arraylike): Value to apply sigmoid to.
        scale (float, optional): Transition width scale of sigmoid. Defaults to 0.001.

    Returns:
        float | arraylike: Sigmoid applied to t.
    """
    return 1/(1+np.exp(-t/scale))


def DT_logp_l_gen(pdt):
    """
    Generates a log-probability function for a numerical time increment distribution. Returned function interpolates a log probability and takes as inputs time increment(s) at which to evaluate as well as (required) time increment coordinates corresponding to the numerical pdf associated with the function. Sped up with numba.
    Args:
        pdt (array-like): The probability density function values for the time increments.
    Returns:
        function: A function that computes the log-probability of a given time increment.
    The returned function, DT_logp, takes the following parameters:
        dt_query (float): The time increment for which the log-probability is to be computed.
        dt (array-like): The array of time increments corresponding to the probability density function values. Must be same length as pdt.
    The DT_logp function computes the log-probability by interpolating the probability density function values and applying sigmoid functions to ensure the values are within the valid range.
    """
    @njit
    def DT_logp(dt_query, dt):
        return np.log(np.interp(dt_query, dt, pdt) * sigmoid(dt_query-dt[0]) * sigmoid(dt[-1]-dt_query))
    return DT_logp

# class AgeModel:
#     def __init__(self, units, geochron, **kwargs):
#         """Initializes the AgeModel class.

#         Args:
#             units (ndarray): nx2 array of unit bottom and top heights for n units.
#             geochron (stratagemc.Geochron): Geochron object containing geochron constraints.
#         """
#         self.units = units
#         self.geochron = geochron
#         # number of units
#         self.n_units = units.shape[0]
#         # thicknesses of units
#         self.thicknesses = np.diff(units, axis=1).squeeze()
#         # confirm that geochron heights are within the section
#         geochron_height_check(units, geochron.h)
#         # trim the section to the top and bottom of the geochron constraints
#         self.units_trim = trim_units(units, geochron.h)
#         # get weights (alpha, beta) for units
#         self.alpha, self.beta = geochron.model_weights(self.units_trim)

#         # create time increment log-probability functions for each pair of constraints
#         DT_logps = []
#         for ii in range(geochron.n_pairs):
#             DT_logps.append(DT_logp_l_gen(geochron.pdts[ii]))
#         self.DT_logps = DT_logps
#         pass

#     def eps_logp(self, dt_val, params):
#         """
#         Generates the log-probability for the deviation from the maximum likelihood time increment values for each pair of constraints. This function is designed to be evaluated at zero (dt_val = 0). Params contains the sedimentation rates and hiatuses for the stratigraphy. For a given params, the corresponding time increments (dt_hat) are computed for each pair. If they are near the true maximum likelihood time increments (geochron.dts_max), then evaluation at dt_val = 0 will return the highest likelihood. If any dt_hat is near zero, then dt_val=0 evaluates near dt=0 (for the true time increment), which has to be small since the time increments must be positive. The log-probability is computed by interpolating the numerical time increment functions and applying a sigmoid function to ensure the values are within a valid range.

#         Args:
#             dt_val (array-like): The observed DT values.
#             params (array-like): Model parameters as concatenation of sedimentation rates and hiatuses.
#         Returns:
#             numpy.ndarray: The log-probability values for each pair of DT values.
#         Notes:
#             - `geochron.n_pairs` is used to determine the number of DT pairs.
#             - The function computes `dt_hat` using sedimentation rates and hiatuses.
#             - The log-probability is calculated by interpolating the observed DT values
#             and applying a sigmoid function to ensure the values are within a valid range.
#         """
#         n = self.geochron.n_pairs
#         # logp = 0
#         logp = np.zeros(n)
#         sed_rates = params[0:self.n_units]
#         hiatuses = params[self.n_units:]
#         # compute dt_hat
#         dt_hat = np.sum(self.alpha/sed_rates.reshape(-1, 1), axis=0) + np.sum(self.beta*hiatuses.reshape(-1, 1), axis=0)
#         for ii in range(n):
#             # shift geochron dt for current dt_hat
#             cur_dt = self.geochron.dts[ii] - dt_hat[ii]
#             logp[ii] = self.DT_logps[ii](dt_val[ii], cur_dt)
#             # logp[ii] = np.log(np.interp(dt_val[ii], cur_dt, geochron.pdts[ii]) * \
#             #                   sigmoid(dt_val[ii]-cur_dt[0]) * \
#             #                   sigmoid(cur_dt[-1]-dt_val[ii]))
#             # logp += cur_logp_fun(t[ii])
#         return logp


def floating_age(sed_rates, hiatuses, units, heights):
    """Floating ages at heights in strat. Assumes t=0 at base of section. Cannot be evaluated at unit contacts. 

    Args:
        sed_rates (arraylike): Sedimentation rates for each unit.
        hiatuses (arraylike): Hiatuses between units. Must be len(sed_rates)-1.
        units (arraylike): nx2 array of unit bottom and top heights for n units.
        heights (arraylike | float): Height(s) at which to evaluate the age model.

    Returns:
        array: Age(s) at the given height(s).
    """
    heights = np.atleast_1d(heights)
    n_heights = len(heights)
    # check that heights are not at contacts
    assert ~np.any(np.isin(heights, units[1:-1])), 'heights cannot be at contacts'
    # floating age model
    times = get_times(sed_rates, hiatuses, units)
    # print(times)
    # evaluate cumulative time at height
    ages = np.zeros(n_heights)
    for ii, height in enumerate(heights):
        # otherwise linearly interpolate sed rate in the unit
        unit_idx = np.argwhere((height >= units[:, 0]) & (height <= units[:, 1]))
        # print(unit_idx)
        ages[ii] = times[unit_idx, 0] + \
            (height - units[unit_idx, 0])/sed_rates[unit_idx]
    return ages.squeeze()


def fit_floating_model(sed_rates, hiatuses, units, geochron, tol=1e-6):
    """Fit a floating age model to geochronologic constraints by maximizing the likelihood of alignment with the geochronologic constraints.

    Args:
        sed_rates (arraylike): Sedimentation rates for each unit.
        hiatuses (arraylike): Hiatuses between units. Must be len(sed_rates)-1.
        units (ndarray): nx2 array of unit bottom and top heights for n units.
        geochron (geochron.Geochron): Geochron object containing geochron constraints.
        tol (float, optional): Tolerance for bounds on optimization. Defaults to 1e-6.

    Returns:
        ndarray: nx2 array of unit bottom and top ages for n units in absolute time
    """
    # floating age model
    geochron_float_ages = floating_age(sed_rates, hiatuses, units, geochron.h)

    def time_offset_cost(offset):
        """
        Cost function to optimize to find the shift in cumulative times that best fits the geochronologic constraints. The goal is to maximize the log likelihood, which is equivalent to minimizing the negative log likelihood.
        Args:
            offset (float): The offset to apply to floating ages; optimization parameter.
        Returns:
            float: The negative log likelihood value.
        """
        # log likelihood
        ll = 0
        for ii in range(geochron.n_constraints):
            ll += np.log(geochron.rv[ii].pdf(geochron_float_ages[ii] + offset))
        return -ll

    # get the offset to optimally align the depth-time history
    offset = minimize_scalar(time_offset_cost,
                             method='bounded',
                             bounds=[geochron.rv[0].ppf(tol),
                                     geochron.rv[0].ppf(1-tol)]).x

    # floating age model
    times = get_times(sed_rates, hiatuses, units)

    return times + offset


def age_depth(units, times):
    """
    Convert units, times arrays to age-depth curve
    Args:
        units (numpy.ndarray): An nx2 array of bottom and top elevations for each of n units.
        times (numpy.ndarray): An nx2 array of bottom and top times for each of n units.
    Returns:
        numpy.ndarray: Vector of ages for each unit.
        numpy.ndarray: Vector of heights for each unit.

    """
    z = units.reshape(-1, order='C')
    t = times.reshape(-1, order='C')
    return t, z


def agemodel_ls(units, geochron,
                sed_rate_bounds=[1e-1, 1e2], hiatus_bounds=[1e-1, 1e3]):
    """Least squares age model fitting.

    Args:
        units (ndarray): nx2 array of unit bottom and top heights for n units.
        geochron (geochron.Geochron): Geochron object containing geochron constraints.
    """
    alpha, beta = geochron.model_weights(units)
    # model matrix
    A = np.vstack((alpha, beta)).T
    d = np.array(geochron.dts_max)

    n_units = units.shape[0]
    n_contacts = n_units - 1

    # bounds on model parameters
    lower_bounds = np.zeros(n_units+n_contacts)
    upper_bounds = np.zeros(n_units+n_contacts)
    # units
    lower_bounds[0:n_units] = sed_rate_bounds[0]
    upper_bounds[0:n_units] = sed_rate_bounds[1]
    # contacts
    lower_bounds[n_units:] = hiatus_bounds[0]
    upper_bounds[n_units:] = hiatus_bounds[1]

    # least squares
    m_bdls = lsq_linear(A, d, bounds=[lower_bounds, upper_bounds])
    m_bdls_sol = m_bdls.x
    # m_bdls_res = m_bdls.fun
    sed_rates = 1/m_bdls_sol[0:n_units]
    hiatuses = m_bdls_sol[n_units:]
    return sed_rates, hiatuses


def agemodel(units, geochron):

    # number of units
    n_units = units.shape[0]
    # thicknesses of units
    thicknesses = np.diff(units, axis=1).squeeze()
    # confirm that geochron heights are within the section
    geochron_height_check(units, geochron.h)
    # trim the section to the top and bottom of the geochron constraints
    units_trim = trim_units(units, geochron.h)
    # get weights (alpha, beta) for units
    alpha, beta = geochron.model_weights(units_trim)

    # create time increment log-probability functions for each pair of constraints
    DT_logps = []
    for ii in range(geochron.n_pairs):
        DT_logps.append(DT_logp_l_gen(geochron.pdts[ii]))
    DT_logps = DT_logps

    def eps_logp(dt_val, params):
        """
        Generates the log-probability for the deviation from the maximum likelihood time increment values for each pair of constraints. This function is designed to be evaluated at zero (dt_val = 0). Params contains the sedimentation rates and hiatuses for the stratigraphy. For a given params, the corresponding time increments (dt_hat) are computed for each pair. If they are near the true maximum likelihood time increments (geochron.dts_max), then evaluation at dt_val = 0 will return the highest likelihood. If any dt_hat is near zero, then dt_val=0 evaluates near dt=0 (for the true time increment), which has to be small since the time increments must be positive. The log-probability is computed by interpolating the numerical time increment functions and applying a sigmoid function to ensure the values are within a valid range.

        Args:
            dt_val (array-like): The observed DT values.
            params (array-like): Model parameters as concatenation of sedimentation rates and hiatuses.
        Returns:
            numpy.ndarray: The log-probability values for each pair of DT values.
        Notes:
            - `geochron.n_pairs` is used to determine the number of DT pairs.
            - The function computes `dt_hat` using sedimentation rates and hiatuses.
            - The log-probability is calculated by interpolating the observed DT values 
            and applying a sigmoid function to ensure the values are within a valid range.
        """
        n = geochron.n_pairs
        # logp = 0
        logp = np.zeros(n)
        sed_rates = params[0:n_units]
        hiatuses = params[n_units:]
        # compute dt_hat
        dt_hat = np.sum(alpha/sed_rates.reshape(-1, 1), axis=0) + \
            np.sum(beta*hiatuses.reshape(-1, 1), axis=0)
        for ii in range(n):
            # shift geochron dt for current dt_hat
            cur_dt = geochron.dts[ii] - dt_hat[ii]
            logp[ii] = DT_logps[ii](dt_val[ii], cur_dt)
            # logp[ii] = np.log(np.interp(dt_val[ii], cur_dt, geochron.pdts[ii]) * \
            #                   sigmoid(dt_val[ii]-cur_dt[0]) * \
            #                   sigmoid(cur_dt[-1]-dt_val[ii]))
            # logp += cur_logp_fun(t[ii])
        return logp

    class LogLike(Op):
        """"blackbox" likelihood class for PyMC which enables use of eps_logp as a likelihood for sampling.

        Args:
            Op (pystensor.graph.Op): PyTensor Op class.
        """

        def make_node(self, dt_val, params) -> Apply:
            dt_val = pt.as_tensor(dt_val)
            params = pt.as_tensor(params)
            inputs = [dt_val, params]
            outputs = [dt_val.type()]
            return Apply(self, inputs, outputs)

        def perform(self, node: Apply,
                    inputs: list[np.ndarray],
                    outputs: list[list[None]]) -> None:
            dt_val, params = inputs
            loglike_eval = eps_logp(dt_val, params)
            outputs[0][0] = np.asarray(loglike_eval)
    return
