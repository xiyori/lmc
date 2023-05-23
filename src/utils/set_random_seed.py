import random
import torch
import numpy as np


def set_random_seed(seed, deterministic = True):
    torch.backends.cudnn.deterministic = deterministic
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
