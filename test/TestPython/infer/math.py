import clr
clr.AddReference("Microsoft.ML.Probabilistic")

from Microsoft.ML.Probabilistic.Math import Rand, Vector, DenseVector, PositiveDefiniteMatrix

def normal(mean, var):
    mean, var = float(mean), float(var)
    return Rand.Normal(mean, var)