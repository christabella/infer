import sys
folder = "../Tests/bin/debug/net461/"
sys.path.append(folder)

import clr
clr.AddReference(folder+"Microsoft.ML.Probabilistic")

from System import Boolean, Double, Int32, Array

from Microsoft.ML.Probabilistic.Models import Variable as ClrVariable

    
class Bernoulli(object):
    """Wraps Microsoft.ML.Probabilistic.Models.Variable.Bernoulli. We should probably
    take care of Beta, Gaussian, etc. in a similar way?
    """
    def __init__(self, prob_true=0.0):
        """
        Args:
        prob_true (float): Has to be float, can't be int.
        """
        self._clr_instance = ClrVariable.Bernoulli(prob_true)

    def __and__(self, other):
        """Similar to how Pyro implemented Gaussian.__add__().
        https://github.com/pyro-ppl/pyro/blob/1f005c8836599fbc2e53d2f729dc6bdd4d840308/pyro/ops/gaussian.py 
        """
        new_variable = Bernoulli()
        new_variable._clr_instance = self._clr_instance.op_BitwiseAnd(self._clr_instance, other._clr_instance)
        return new_variable

    def observed(self, val):
        self._clr_instance.ObservedValue = val

