
# Temporary hack as Visual Studio Python environment is located in
# 'C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Python37_64'
# not this directory, so infer/ package is not in sys.path. 
import sys
sys.path.append('C:\\Users\\Administrator\\Source\\Repos\\infer\\test\\TestPython')

import infer as inf
from infer.variables import Bernoulli

def two_coins():
    first_coin = Bernoulli(0.5)
    second_coin = Bernoulli(0.5)
    both_heads = first_coin & second_coin
    engine = inf.Engine()
    print(f"Probability both coins are heads: {engine.infer(both_heads)}")
    both_heads.observed(False)
    print(f"Probability distribution over first coin: {engine.infer(first_coin)}")

if __name__ == '__main__':
    two_coins()
