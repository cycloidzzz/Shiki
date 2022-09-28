from typing import Dict, Tuple, Optional
import torchvision
import torch
import numpy as np
from torchvision import transforms

import shiki as sk
import shiki.autograd as tgd

def round_up(n : int, d : int) -> int:
    if n % d == 0:
        return n // d
    else:
        return n // d + 1

def preprocess_dataset(dataset : torch.utils.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    #features, labels = dataset.data
    features : torch.tensor = dataset.data
    labels :torch.tensor = dataset.targets

    features = features / 255
    num_samples : int = features.size()[0]
    features = features.reshape(num_samples, -1)
    features : np.ndarray = features.numpy()
    labels : np.ndarray = labels.numpy()

    return (features, labels)

def mnist_dataset(download : bool):
    train_data = torchvision.datasets.MNIST(
        root='./data/MNIST',
        train = True,
        transform = transforms.ToTensor(),
        download = download
    )

    test_data = torchvision.datasets.MNIST(
        root='./data/MNIST',
        train = False,
        transform = transforms.ToTensor(),
        download = download
    )

    return train_data, test_data

def softmax_classification(num_epochs : int,
                           batch_size : int,
                            train_data : torch.utils.data.Dataset,
                            test_data : torch.utils.data.Dataset,
                            lr : float = 1e-3):
    train_features, train_labels = preprocess_dataset(train_data)
    test_features, test_labels = preprocess_dataset(test_data)

    # Static Graph definition.
    weight = tgd.Variable(name="weight")
    bias = tgd.Variable(name="bias")
    x = tgd.Variable(name="x")
    labels = tgd.Variable(name="labels")
    logits = tgd.matmul_op(x, weight) + bias
    loss = tgd.softmax_cross_entropy_with_logits_op(logits=logits, labels=labels)

    # Model Parameters.
    weight_v = np.random.randn(28 * 28, 10)
    bias_v = np.random.uniform(-1, 1, size=(10,))

    g_w, g_b, g_x, g_l = tgd.gradient(loss, [weight, bias, x, labels])
    executor = tgd.Executor([logits, loss, g_w, g_b])

    num_train : int = train_features.shape[0]
    num_test : int = test_features.shape[0]

    num_train_batch : int = round_up(num_train, batch_size)
    num_test_batch : int = round_up(num_test, batch_size)

    for epoch in range(num_epochs):
        batch_index_list : np.ndarray = np.arange(num_train)
        test_index_list : np.ndarray = np.arange(num_test)
        np.random.shuffle(batch_index_list)

        train_acc : int = 0
        test_acc : float = 0
        for i in range(num_train_batch):
            start_idx : int = i * batch_size
            end_idx : int = min(start_idx + batch_size, num_train)

            batch_idx = batch_index_list[start_idx : end_idx]

            features_v = train_features[batch_idx]
            index_v = train_labels[batch_idx]
            labels_v = np.eye(10)[index_v]

            logits_v, loss_v, g_w_v, g_b_v = executor.run(
                feed_dict = {
                    weight : weight_v, 
                    bias : bias_v, 
                    x : features_v, 
                    labels : labels_v
            })

            weight_v -= lr * g_w_v
            #TODO (cycloidz) : Support 'broadcast' arithmetic operations.
            bias_v -= lr * np.sum(g_b_v, axis=0)

            pred_index_v : np.ndarray = np.argmax(logits_v, axis=-1)

            train_acc += np.equal(pred_index_v, index_v).sum()

        for i in range(num_test_batch):
            start_idx : int = i * batch_size
            end_idx : int = min(start_idx + batch_size, num_test)

            batch_idx = test_index_list[start_idx : end_idx]

            f_v = test_features[batch_idx]
            index_v = test_labels[batch_idx]
            l_v = np.eye(10)[index_v]

            logits_v, loss_v, _, _ = executor.run(
                feed_dict = {
                    weight : weight_v,
                    bias : bias_v,
                    x : f_v,
                    labels : l_v
                }
            )

            pred_index_v : np.ndarray = np.argmax(logits_v, axis=-1)
            test_acc += np.equal(pred_index_v, index_v).sum()

        train_acc = train_acc / num_train
        test_acc = test_acc / num_test
        
        print(f"In epoch {epoch}, train accuracy = {train_acc}, test accuracy = {test_acc}")



if __name__ == "__main__":
    train_data, test_data = mnist_dataset(download = True)
    softmax_classification(
        num_epochs=100,
        batch_size=32,
        train_data=train_data,
        test_data=test_data
    )