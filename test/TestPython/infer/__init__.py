# TODO: Load paths to DLL's in a better way, e.g.
# https://github.com/QuantConnect/Lean/blob/7f941edeeca3b1a1b95aab0bff04db388a24df63/Common/Python/PythonInitializer.cs#L55 
import sys
folder = "../Tests/bin/debug/net461/"
sys.path.append(folder)

from .common import Engine, constrain

__all__ = ["Engine"]