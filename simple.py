import numpy as np
import shiki as sk
import shiki.autograd as autograd

if __name__ == "__main__":
    x = autograd.Variable("x")
    y = autograd.Variable("y")
    z = 2 * x + y
    print(z)
