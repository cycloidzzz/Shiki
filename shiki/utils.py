import numpy as np
from typing import Union

def is_scalar_type(x : Union[int, float, np.ndarray]) -> bool:
    return isinstance(x, int) or isinstance(x, float) or isinstance(x, np.ndarray)