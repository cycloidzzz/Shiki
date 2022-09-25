import numpy as np
import shiki as sk
import shiki.autograd as autograd

if __name__ == "__main__":
    x = autograd.Variable("x")
    y = autograd.Variable("y")
    z = 2 * x + y

    grad_x, grad_y = autograd.gradient(z, [x, y])

    session : autograd.Executor = autograd.Executor(
        [z, grad_x, grad_y]
    ) 

    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)

    z_val, x_grad_v, y_grad_v = session.run(feed_dict = {
        x : x_val,
        y : y_val
    })

    print(z_val)