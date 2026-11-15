"""one shared random generator, so 'anvil.manual_seed(0)' makes runs reproducible"""

import numpy as np

_rng=np.random.default_rng()

def manual_seed(seed):
    global _rng
    _rng=np.random.default_rng(seed)


def get_rng():
    return _rng
