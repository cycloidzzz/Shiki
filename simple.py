import numpy as np
import shiki as sk
import shiki.autograd as autograd

def test_mul_add():
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


def test_softmax():
    x = autograd.Variable("x")
    y = autograd.softmax_op(x)

    grad_x, = autograd.gradient(y, [x])

    session : autograd.Executor = autograd.Executor([y, grad_x])

    x_val = np.random.randn(3, 4)
    y_val , grad_x_v = session.run(feed_dict = {x : x_val})

    print(f"x_val = {x_val}, y_val = {y_val}")
    print(f"gradient of x_val = {grad_x_v}")


if __name__ == "__main__":
    test_mul_add()
    test_softmax()