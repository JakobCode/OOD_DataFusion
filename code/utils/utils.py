"""
Utility functions used in multiple scripts. 
"""

import random
import numpy as np
import torch

def set_seed(seed):
    """Sets seed value."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
