import clr
clr.AddReference("Microsoft.ML.Probabilistic")
clr.AddReference("Microsoft.ML.Probabilistic.Compiler")

from System import Boolean, Double, Int32, Array

from Microsoft.ML.Probabilistic.Models import InferenceEngine as ClrInferenceEngine, \
                                              Variable as ClrVariable
from Microsoft.ML.Probabilistic.Compiler.Reflection import Invoker

class Engine:
    def __init__(self):
        self._clr_instance = ClrInferenceEngine()

    def infer(self, variable):
        return self._clr_instance.Infer(variable._clr_instance)


def constrain(var):
    ClrVariable.ConstrainTrue(var._clr_instance)