import sys
folder = "../Tests/bin/debug/net461/"
sys.path.append(folder)

import clr
clr.AddReference(folder+"Microsoft.ML.Probabilistic")
clr.AddReference(folder+"Microsoft.ML.Probabilistic.Compiler")

from System import Boolean, Double, Int32, Array

from Microsoft.ML.Probabilistic.Models import InferenceEngine as ClrInferenceEngine
from Microsoft.ML.Probabilistic.Compiler.Reflection import Invoker

class Engine:
    def __init__(self):
        self._clr_instance = ClrInferenceEngine()

    def infer(self, variable):
        return self._clr_instance.Infer(variable._clr_instance)