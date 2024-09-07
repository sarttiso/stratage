import logging
import sys

import numpy as np
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
    assert np.all(geochron_heights >= np.min(unit_heights)), 'geochron heights are below section'
    assert np.all(geochron_heights <= np.max(unit_heights)), 'geochron heights are above section'
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


