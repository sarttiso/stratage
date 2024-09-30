import logging
import sys

import numpy as np

from numba import njit

import pymc as pm
import pytensor.tensor as pt
from pytensor.graph import Apply, Op

from scipy.optimize import minimize_scalar, lsq_linear

from .geochron import Geochron

from stratage import __version__

__author__ = "Adrian Tasistro-Hart"
__copyright__ = "Adrian Tasistro-Hart"
__license__ = "GPL-3.0-only"

_logger = logging.getLogger(__name__)


def geochron_height_check(unit_heights, geochron_heights):
    """
    Ensure that geochron heights are within units and not at contacts. Also ensure that geochron heights are within the section. Run before trimming section to the top and bottom of the geochron constraints.

    Args:
        unit_heights (numpy.ndarray): Bottom and top heights of units in section. 
            2d array (nx2) of bottom and top heights for each of n units
        geochron_heights (arraylike): Heights of geochron constraints in the section

    Raises:
        AssertionError: If any of the geochron heights are below the section or above the section.
        AssertionError: If any of the geochron heights are at contacts.
    """
    # check that geochron heights are within the section
    assert np.all(geochron_heights >= np.min(unit_heights)
                  ), 'geochron heights are below section'
    assert np.all(geochron_heights <= np.max(unit_heights)
                  ), 'geochron heights are above section'
    # check that geochron heights are not at contacts except for the top and bottom
    offending_heights = np.intersect1d(geochron_heights, np.unique(unit_heights[1:-1]))
    assert len(offending_heights) == 0, \
        f'geochron heights cannot be at contacts, {offending_heights}'


def trim_units(unit_heights, geochron_heights):
    """
    Trim the top and bottom of the section to the top and bottom of the geochron constraints. Run after geochron_height_check.

    Args:
        unit_heights (numpy.ndarray): Bottom and top heights of units in section. 
            2d array-like (nx2) of bottom and top heights for each of n units
        geochron_heights (arraylike): Heights of geochron constraints in section

    Returns:
        numpy.ndarray: Trimmed unit heights after adjusting for the top and bottom units

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
        sed_rates (arraylike): Sedimentation rates for each unit
        hiatuses (arraylike): Hiatuses between units
        units (numpy.ndarray): Bottom and top heights of units in section. 
            2d array (nx2) of bottom and top heights for each of n units

    Returns:
        times (numpy.ndarray): Floating times array for units, same shape as units
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
        numba.core.registry.CPUDispatcher: A function that computes the log-probability of a given time increment.
            The returned function, DT_logp, takes the following parameters:
                dt_query (float): The time increment for which the log-probability is to be computed.
                dt (array-like): The array of time increments corresponding to the probability density function values. Must be same length as pdt.
            The DT_logp function computes the log-probability by interpolating the probability density function values and applying sigmoid functions to ensure the values are within the valid range.
    """
    @njit
    def DT_logp(dt_query, dt):
        return np.log(np.interp(dt_query, dt, pdt) * sigmoid(dt_query-dt[0]) * sigmoid(dt[-1]-dt_query))
    return DT_logp


def randlike_gen(geochron, units):
    """
    Generate function for generating random draws from likelihood. Permits generating prior_predictive samples from a model with the CustomDist likelihood.

    Args:
        geochron (geochron.Geochron): Geochron object containing geochronologic constraints.
        units (numpy.ndarray): Bottom and top heights of units in section. 
            2d array (nx2) of bottom and top heights for each of n units

    Returns:
        function: Function for generating random draws from likelihood.
            The returned function, eps_rand, takes the following parameters:
                | params (array-like): Array of parameters; concatenation of sedimentation rates and hiatuses.
                | rng (numpy.random.Generator): Random number generator.
                | size (int or tuple of ints, optional): Output shape. Default is None. Last dimension must be the number of pairs of geochron constraints.
            The function returns a numpy.ndarray of random draws.
    """
    n_units = units.shape[0]
    alpha, beta = geochron.model_weights(units)

    def eps_rand(params, rng, size):
        """
        Generate random samples from likelihood. Params is the concatenation of sedimentation rates and hiatuses.

        Args:
            params (array-like): Array of parameters; concatenation of sedimentation rates and hiatuses.
            rng (numpy.random.Generator): Random number generator.
            size (int or tuple of ints, optional): Output shape. Default is None. Last dimension must be the number of pairs of geochron constraints.

        Returns:
            numpy.ndarray: Random draws.

        Raises:
            AssertionError: If size is incompatible with the number of pairs.

        Notes:
            - The function calculates the random draws based on the given parameters and geochron constraints.
            - If size is not provided or is None, it will default to the number of pairs.
            - If there is only one pair, the geochron dt will be shifted for the current dt_max.
            - If there are multiple pairs, the geochron dt for each pair will be shifted for the corresponding dt_max.
        """
        sed_rates = params[0:n_units]
        hiatuses = params[n_units:]
        dt_hat = np.sum(alpha/sed_rates.reshape(-1, 1), axis=0) + \
            np.sum(beta*hiatuses.reshape(-1, 1), axis=0)

        if size is None or not size:
            size = geochron.n_pairs
        size = np.atleast_1d(size)
        # print(size)
        if geochron.n_pairs != 1:
            assert size[-1] == geochron.n_pairs, 'size incompatible with ys, DTs'
        if geochron.n_pairs == 1:
            # shift geochron dt for current dt_max
            cur_dt = geochron.dts[0] - dt_hat
            CDF = np.cumsum(geochron.pdts[0]) * np.mean(np.diff(cur_dt))
            return np.interp(rng.uniform(size=size), CDF, cur_dt)
        else:
            rand = []
            for ii in range(geochron.n_pairs):
                # shift geochron dt for current dt_max
                cur_dt = geochron.dts[ii] - dt_hat[ii]
                CDF = np.cumsum(geochron.pdts[ii]) * np.mean(np.diff(cur_dt))
                rand.append(np.interp(rng.uniform(size=size[0:-1]), CDF, cur_dt))
            return np.stack(rand, axis=len(size)-1)
    return eps_rand


def loglike_gen(geochron, units):
    """Returns log-likelihood object for use in PyMC.CustomDist as a likelihood for posterior sampling. See subfunction eps_logp for details on the log-likelihood computation.

    Args:
        geochron (geochron.Geochron): Geochron object containing geochronologic constraints.
        units (ndarray): nx2 array of unit bottom and top heights for n units.

    Returns:
        LogLike: PyTensor Op class for likelihood evaluation.
    """
    # model weights
    alpha, beta = geochron.model_weights(units)
    n_units = units.shape[0]
    # time increment log-probability functions
    DT_logps = []
    for ii in range(geochron.n_pairs):
        DT_logps.append(DT_logp_l_gen(geochron.pdts[ii]))

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

    loglike_op = LogLike()
    return loglike_op


def floating_age(sed_rates, hiatuses, units, heights):
    """Floating ages at heights in strat. Assumes t=0 at base of section. Times at contact heights are returned as the age of the top of the unit below the contact.

    Args:
        sed_rates (arraylike): Sedimentation rates for each unit.
        hiatuses (arraylike): Hiatuses between units. Must be len(sed_rates)-1.
        units (arraylike): nx2 array of unit bottom and top heights for n units.
        heights (arraylike | float): Height(s) at which to evaluate the age model.

    Returns:
        array: Age(s) at the given height(s).
    """
    # floating age model
    times = get_times(sed_rates, hiatuses, units)

    return age(times, units, heights)


def age(times, units, heights):
    """Interpolate time at given heights in strat. Times at contact heights are returned as the age of the top of the unit below the contact.

    Args:
        times (arraylike): nx2 array of unit bottom and top times for n units.
        units (arraylike): nx2 array of unit bottom and top heights for n units.
        heights (arraylike): Height(s) at which to evaluate the age model.

    Returns:
        array: Age(s) at the given height(s).
    """
    heights = np.atleast_1d(heights)
    n_heights = len(heights)
    # confirm that heights are within units
    assert np.all((heights >= units[0, 0]) & (
        heights <= units[-1, 1])), 'heights must be within units'
    # evaluate cumulative time at height
    ages = np.zeros(n_heights)
    # indices into units and times of each height
    idxs = np.argmax((heights.reshape(-1, 1) >= units[:, 0]) &
                     (heights.reshape(-1, 1) <= units[:, 1]), axis=1)
    idxs = np.atleast_1d(idxs)
    for ii, height in enumerate(heights):
        # otherwise linearly interpolate sed rate in the unit
        ages[ii] = np.interp(height, units[idxs[ii], :], times[idxs[ii], :])
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
    # check that geochron heights are not at contacts
    assert ~np.any(np.isin(geochron.h, units[1:-1])), 'heights cannot be at contacts'

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


def model_ls(units, geochron,
             sed_rate_bounds=None,
             hiatus_bounds=None):
    """Least squares age model fitting.

    Args:
        units (ndarray): nx2 array of unit bottom and top heights for n units.
        geochron (geochron.Geochron): Geochron object containing geochron constraints.
        sed_rate_bounds (list, optional): Bounds on sedimentation rates. Defaults to None.
        hiatus_bounds (list, optional): Bounds on hiatuses. Defaults to None.

    Returns:
        sed_rates (numpy.ndarray): Sedimentation rates for each unit.
        hiatuses (numpy.ndarray): Hiatuses between units.
    """
    alpha, beta = geochron.model_weights(units)
    # model matrix
    A = np.vstack((alpha, beta)).T
    d = np.array(geochron.dts_max)

    n_units = units.shape[0]
    n_contacts = n_units - 1

    # set defaults for bounds
    if sed_rate_bounds is None:
        sed_rate_bounds = [1e-1, 1e2]
    if hiatus_bounds is None:
        hiatus_bounds = [1e-1, 1e3]

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


def model(units, geochron, sed_rates_prior, hiatuses_prior,
          draws=1000, **kwargs):
    """MCMC modeling of sed rates.

    User must provide priors for sedimentation rates and hiatuses, which are functions valid as dist arguments to pymc.CustomDist(dist=dist). In this case, the only valid signature is dist(size=size), since this function does not permit additional arguments to the distribution.

    Args:
        units (ndarray): nx2 array of unit bottom and top heights for n units.
        geochron (geochron.Geochron): Geochron object containing geochron constraints.
        sed_rates_prior (function): Prior distribution for sedimentation rates. Must be valid as dist argument to pymc.CustomDist(dist=dist). Signature is sed_rate_prior(size=size).
        hiatuses_prior (function): Prior distribution for hiatuses. Must be valid as dist argument to pymc.CustomDist(dist=dist). Signature is hiatus_prior(size=size).
        draws (int, optional): Number of MCMC draws. Defaults to 1000.
        **kwargs: Additional keyword arguments to pass to the MCMC sampler.

    Returns:    
        arviz.InferenceData: ArviZ InferenceData object containing the MCMC trace.
    """

    # number of units
    n_units = units.shape[0]
    # number of contacts
    n_contacts = n_units - 1
    # confirm that geochron heights are within the section
    geochron_height_check(units, geochron.h)
    # trim the section to the top and bottom of the geochron constraints
    units_trim = trim_units(units, geochron.h)
    # create least squares model as initial guess
    sed_rates_ls, hiatuses_ls = model_ls(units, geochron)
    # create time increment log-like function
    loglike_op = loglike_gen(geochron, units_trim)
    # create model
    coords = {'units': np.arange(n_units),
              'contacts': np.arange(n_contacts),
              'pairs': np.arange(geochron.n_pairs)}
    model = pm.Model(coords=coords)
    with model:
        # sed rates
        sed_rates = pm.CustomDist('sed_rates',
                                  dist=sed_rates_prior,
                                  shape=(n_units,),
                                  dims='units')
        model.set_initval(sed_rates, sed_rates_ls)
        # hiatuses
        hiatuses = pm.CustomDist('hiatuses',
                                 dist=hiatuses_prior,
                                 shape=(n_contacts,),
                                 dims='contacts')
        model.set_initval(hiatuses, hiatuses_ls)
        # likelihood
        likelihood = pm.CustomDist('likelihood',
                                   pm.math.concatenate([sed_rates, hiatuses]),
                                   observed=np.zeros(geochron.n_pairs),
                                   logp=loglike_op)
    # sample
    vars_list = list(model.values_to_rvs.keys())[:-1]
    with model:
        trace = pm.sample(draws=draws, **kwargs)
    return trace


def model2ages(trace, n_posterior=None):
    # if n_posterior is None, use effective sample size to dictate number of posterior samples
    if n_posterior is None:
        # n_posterior = trace.n_eff
        return
