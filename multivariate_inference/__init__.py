# -*- coding: utf-8 -*-

"""Top-level package for multivariate_inference."""

from __future__ import absolute_import

__author__ = """Eshin Jolly"""
__email__ = 'eshin.jolly.gr@dartmouth.edu'
__version__ = '0.1.0'

__all__ = ["helpers"]

from .helpers import (upper,
                      isPSD,
                      nearestPSD,
                      easy_multivariate_normal,
                      kde_pvalue,
                      create_heterogeneous_simulation
                      )
from .dependence import (double_center,
                         u_center,
                         distance_correlation,
                         procrustes_similarity
                         )
