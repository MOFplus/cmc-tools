import pytest
from weaver.RTA.genetic import Environment

def score(chrome, dummy):
    return sum(chrome)

def test_genetic():
    env = Environment(size=50, maxgenerations=100, optimum=0, score_func=score, alleles = 10*[4])
    env.run()
