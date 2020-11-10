"""Interface for int and enum factors.
https://dotnet.github.io/infer/userguide/Int%20factors.html
"""

import sys
folder = "../Tests/bin/debug/net461/"
sys.path.append(folder)

import clr
clr.AddReference(folder+"Microsoft.ML.Probabilistic")

from System import Boolean, Double, Int32, Array

from Microsoft.ML.Probabilistic.Models import Variable as ClrVariable
from typing import Union

    
class Discrete():
    """Wraps Microsoft.ML.Probabilistic.Models.Variable.Bernoulli. We should probably
    take care of Beta, Gaussian, etc. in a similar way?
    """
    def __init__(self):
        pass