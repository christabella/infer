"""https://dotnet.github.io/infer/userguide/Double%20factors.html
"""
import clr
clr.AddReference("Microsoft.ML.Probabilistic")

from System import Boolean, Double, Int32, Array

from Microsoft.ML.Probabilistic.Models import Variable as ClrVariable
from typing import Union

from .base import Variable

class Gaussian(Variable):
    def __init__(self, mean=None, variance=None, shape=1):
        """Create a Gaussian variable of the given shape.
        """
        try:
            mean, variance = float(mean), float(variance)
            self._clr_instance = ClrVariable.GaussianFromMeanAndVariance(mean, variance)
        except:
            pass

class Gamma():
    pass

class Wishart():
    pass