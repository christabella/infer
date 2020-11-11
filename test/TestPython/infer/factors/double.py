"""https://dotnet.github.io/infer/userguide/Double%20factors.html
"""
import clr
clr.AddReference("Microsoft.ML.Probabilistic")

from System import Boolean, Double, Int32, Array

from Microsoft.ML.Probabilistic.Models import Variable as ClrVariable, Range as ClrRange, VariableArray
from Microsoft.ML.Probabilistic.Factors import Factor as ClrFactor
from typing import Union

from .base import Variable, _clr_if_var

class Gaussian(Variable):
    def __init__(self, mean=None, variance=None, precision=None, shape=1):
        """Create a Gaussian variable of the given shape.

        Args:
            mean: Specified either as a constant (number) or a prior (Variable).
            variance: Similarly either a number or a Variable.
            precision: Either precision or variance should be specified.
            shape: An int or tuple of sizes per dimension.
        """
        # If not given mean and variance, e.g. when creating an unset Variable
        # during operator overload, don't create a ClrVariable instance.
        if mean == None:
            return
        self._mean = _clr_if_var(mean)
        # Determine if either variance or precision is specified.
        if precision == None:
            self._clr_class = ClrVariable.GaussianFromMeanAndVariance
            self._var_or_prec = _clr_if_var(variance)
        elif variance == None:
            self._clr_class = ClrVariable.GaussianFromMeanAndPrecision
            self._var_or_prec = _clr_if_var(precision)
        # Determine if single, array, or 2D array, or jagged array.
        if shape == 1:
            self._init_single()
        elif isinstance(shape, int):
            self._init_array(shape)

    def _init_single(self):
        self._clr_instance = self._clr_class(self._mean, self._var_or_prec)

    def _init_array(self, shape):
        range = ClrRange(shape)
        gaussians = self._clr_class(self._mean, self._var_or_prec).ForEach(range)
        vars = ClrVariable.Array[Double](range)
        vars.set_Item(range, gaussians)
        self._clr_instance = vars

class Gamma(Variable):
    def __init__(self, form=None, scale=None, shape=1):
        """Create a Gamma variable of the given shape.

        Args:
          form: Also known as the shape parameter of the gamma distribution.
          scale: Scala parameter of the gamma distribution.
          shape: Size of the array.
        """
        try:
            form, scale = float(form), float(scale)
            self._clr_instance = ClrVariable.GammaFromShapeAndScale(form, scale)
        except:
            if isinstance(form, Variable):
                return NotImplementedError        

class Wishart():
    pass