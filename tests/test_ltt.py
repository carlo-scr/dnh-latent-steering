import numpy as np

from dnh.ltt import binom_pvalue, calibrate, fixed_sequence, min_n_zero_losses, paired_harm
from dnh.steering import select


def test_min_n():
    assert min_n_zero_losses(0.05, 0.1) == 45


def test_fixed_sequence_stops_at_first_failure():
    assert fixed_sequence([0.01, 0.02, 0.5, 0.01], 0.1) == 1
    assert fixed_sequence([0.5], 0.1) == -1


def test_paired_harm():
    assert paired_harm([1, 1, 0, 0], [1, 0, 1, 0]).tolist() == [0, 1, 0, 0]


def test_ltt_validity_monte_carlo():
    """Under a known risk curve, P[certified lam has risk > eps] <= delta."""
    rng = np.random.default_rng(0)
    lams = [0, 1, 2, 4, 8, 16]
    true_harm = {0: 0.0, 1: 0.01, 2: 0.03, 4: 0.06, 8: 0.12, 16: 0.03}  # non-monotone on purpose
    eps, delta, n, bad = 0.05, 0.1, 200, 0
    trials = 2000
    for _ in range(trials):
        harm = {(l, np.inf): rng.random(n) < true_harm[l] for l in lams}
        abst = {(l, np.inf): np.zeros(n, int) for l in lams}
        res = calibrate(lams, [np.inf], harm, abst, eps=eps, alpha=0.1, delta=delta)
        bad += true_harm[res["lam"]] > eps
    assert bad / trials <= delta


def test_select_lambda_zero_is_base_and_inf_is_argmin():
    s = np.array([[3.0, 1.0, 2.0]])
    assert select(s, 0.0)[0] == 0
    assert select(s, np.inf)[0] == 1
    assert select(s, 0.0, eta=0.5)[0] == -1  # nothing certified -> abstain
    assert select(s, 4, family="bestofk")[0] == 1
    assert select(s, 1, family="bestofk")[0] == 0


def test_binom_pvalue_monotone():
    assert binom_pvalue(0, 100, 0.05) < binom_pvalue(5, 100, 0.05)
