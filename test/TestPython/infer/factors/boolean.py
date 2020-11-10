"""Interface for boolean factors.
https://dotnet.github.io/infer/userguide/Bool%20factors.html
"""
import clr
clr.AddReference("Microsoft.ML.Probabilistic")

from System import Boolean, Double, Int32, Array

from Microsoft.ML.Probabilistic.Models import Variable as ClrVariable
from typing import Union

from .base import Variable
    
class Bernoulli(Variable):
    """Wraps Microsoft.ML.Probabilistic.Models.Variable.Bernoulli. We should probably
    take care of Beta, Gaussian, etc. in a similar way?
    """
    def __init__(self, prob_true: Union[int, float] = 0.0):
        """
        Args:
          prob_true: Probability of the binary variable being true.
        """
        prob_true = float(prob_true)
        self._clr_instance = ClrVariable.Bernoulli(prob_true)

