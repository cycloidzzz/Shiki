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


def test_expand_squeeze():
    x = autograd.Variable("x")
    y = autograd.expand_dims_op(x, axis=1)

    grad_x, = autograd.gradient(y, [x])

    executor = autograd.Executor([y, grad_x])

    x_val = np.random.randn(3,3)
    y_val, grad_x_val = executor.run(
        feed_dict = {x : x_val}
    )

    print(f"x_val = {x_val}, y_val = {y_val}")
    print(f"gradient of x_val = {grad_x_val}")


def test_softmax_cross_entropy_with_logits():

    x = autograd.Variable("x")
    y = autograd.Variable("y")
    loss = autograd.softmax_cross_entropy_with_logits_op(logits=x, labels=y)

    grad_x, grad_y = autograd.gradient(loss, [x, y])

    x_v = np.random.randn(3, 4)
    y_s = np.random.randint(low=0,high=4,size=(3,))
    y_v = np.eye(4)[y_s]

    executor = autograd.Executor([loss, grad_x])
    loss_val, grad_x_val = executor.run(
        feed_dict = {x : x_v, y : y_v}
    )

    print(f"x_v = {x_v}, y_v = {y_v}")
    print(f"loss_val = {loss_val}")
    print(f"gradient of x_val = {grad_x_val}")
        

if __name__ == "__main__":
    test_mul_add()
    test_softmax()
    test_expand_squeeze()
    test_softmax_cross_entropy_with_logits()