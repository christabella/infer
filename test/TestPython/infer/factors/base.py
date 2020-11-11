import clr
clr.AddReference("Microsoft.ML.Probabilistic")
from Microsoft.ML.Probabilistic.Models import Variable as ClrVariable, InferenceEngine, Range as ClrRange, VariableArray

from System import Boolean, Double, Int32, Array


class Variable():
    """Base class."""
    def __init__(self, shape=1, dtype=float):
        # TODO support other shapes and types of variables.
        _dtype_map = {float: Double,
                      int: Int32}
        _dtype = _dtype_map[dtype]
        if shape == 1:
            self._clr_instance = ClrVariable.New[_dtype]()
        elif isinstance(shape, int):
            # TODO think about supporting .Named()
            self._clr_instance = ClrVariable.Array[_dtype](ClrRange(shape))
        elif isinstance(shape, tuple):
            return NotImplementedError

    def observed(self, val):
        self._clr_instance.ObservedValue = val

    def __and__(self, other):
        """Create new Variable of the same class.
        
        Note: Similar to how Pyro implemented Gaussian.__add__().
        https://github.com/pyro-ppl/pyro/blob/1f005c8836599fbc2e53d2f729dc6bdd4d840308/pyro/ops/gaussian.py 
        """
        new_var = type(self)()
        new_var._clr_instance = self._clr_instance.op_BitwiseAnd(
            self._clr_instance, other._clr_instance)
        return new_var

    def __or__(self, other):
        raise NotImplementedError


    def __gt__(self, other):
        new_var = type(self)()
        new_var._clr_instance = self._clr_instance.op_GreaterThan(
            self._clr_instance, other._clr_instance)
        return new_var


def _clr_if_var(const_or_var):
    """Return CLR instance if Variable, else float number.

    Used in factors (subclasses of Variable) which follow either a constant or prior distribution.
    """
    if isinstance(const_or_var, Variable):
        return const_or_var._clr_instance
    return float(const_or_var)