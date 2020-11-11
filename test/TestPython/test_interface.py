# Temporary hack to import infer, as Visual Studio Python environment is located in
# 'C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Python37_64'
# not this directory, so infer/ package is not in sys.path. 
import sys
sys.path.append('C:\\Users\\Administrator\\Source\\Repos\\infer\\test\\TestPython')
import infer as inf
from infer.factors import Bernoulli, Gamma, Gaussian, Variable

def two_coins():
    first_coin = Bernoulli(0.5)
    second_coin = Bernoulli(0.5)
    both_heads = first_coin & second_coin
    engine = inf.Engine()
    both_heads_inferred = engine.infer(both_heads)  # Microsoft.ML.Probabilistic.Distributions.Bernoullli object
    print(f"Probability both coins are heads: {engine.infer(both_heads)}")
    assert(str(both_heads_inferred) == "Bernoulli(0.25)")
    assert(both_heads_inferred.GetMean() == 1/4)

    both_heads.observed(False)
    first_coin_inferred = engine.infer(first_coin)
    print(f"Probability distribution over first coin: {first_coin_inferred}")
    assert(str(first_coin_inferred) == "Bernoulli(0.3333)")
    assert(first_coin_inferred.GetMean() == 1/3)

def truncated_gaussian():
    x = Gaussian(0, 1)
    threshold = Variable()  # Unset scalar variable, maybe rename.
    inf.constrain(x > threshold)
    engine = inf.Engine()
    for thresh in [i*0.1 for i in range(11)]:
        threshold.observed(thresh)
        print(f"Dist over x given thresh of {thresh} = {engine.infer(x)}")

def learning_a_gaussian_range():
    n = 100
    data = [inf.math.normal(0, 1) for i in range(n)]

    mean = Gaussian(0, 100)
    precision = Gamma(1, 1)
    x = Gaussian(mean, precision=precision, shape=n)
    x.observed(data)
    engine = inf.Engine()
    print(f"mean={engine.infer(mean)}")
    print(f"precision={engine.infer(precision)}")

def mixture_of_gaussians():
    k = 2  # Number of mixture components.
    d = 2  # Data is 2-dimensional.
    # Create an array of k d-dimensional Gaussian variables as the means of the components.
    mean = Normal([0]*d, [1]*d, sample_shape=k) 
    means = variable

if __name__ == '__main__':
    two_coins()
    truncated_gaussian()
    learning_a_gaussian_range()