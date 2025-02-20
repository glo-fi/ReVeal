import sys, os
sys.path.insert(1, os.getcwd())
import torch
from model.metriclearner import MetricLearningModel
from torch.optim import Adam
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    torch.manual_seed(1000)
    batch_size = 128
    input_dim = 200
    hdim = 256
    x_a = torch.randn(size=[batch_size+32, input_dim])
    test_x = x_a[batch_size:, :]
    x_a = x_a[:batch_size, :]
    targets = torch.randint(0, 2, size=[batch_size + 32])
    test_y = targets[batch_size:]
    targets = targets[:batch_size]
    x_p = torch.randn(size=[batch_size, input_dim])
    x_n = torch.randn(size=[batch_size, input_dim])

    model = MetricLearningModel(input_dim=input_dim, hidden_dim=hdim)
    optimizer = Adam(model.parameters())

    for epoch in range(50):
        model.zero_grad()
        optimizer.zero_grad()
        prediction_prob, representation, batch_loss = model(
            example_batch=x_a,
            targets=targets,
            positive_batch=x_p,
            negative_batch=x_n)
        repr = representation.detach().cpu().numpy()
        prediction_classes = np.argmax(prediction_prob.detach().cpu().numpy(), axis=-1)
        # print(
        #     "Epoch %3d, Loss: %10.4f, Accuracy: %5.2f, Precision: %5.2f, Recall: %5.2f, F1: %5.2f" % (
        #         epoch, batch_loss.detach().cpu().item(),
        #         acc(targets, prediction_classes), pr(targets, prediction_classes),
        #         rc(targets, prediction_classes), f1(targets, prediction_classes)
        #     )
        # )
        if epoch % 1 == 0:
            prediction_prob, representation, batch_loss = model(
                example_batch=test_x,
                targets=test_y)
            repr = representation.detach().cpu().numpy()
            prediction_classes = np.argmax(prediction_prob.detach().cpu().numpy(), axis=-1)
            print('=' * 100)
            print(
                "Test  %3d, Loss: %10.4f, Accuracy: %5.2f, Precision: %5.2f, Recall: %5.2f, F1: %5.2f" % (
                    epoch, batch_loss.detach().cpu().item(),
                    accuracy_score(test_y.numpy(), prediction_classes), precision_score(test_y.numpy(), prediction_classes),
                    recall_score(test_y.numpy(), prediction_classes), f1_score(test_y.numpy(), prediction_classes)
                )
            )
            print('=' * 100)
        batch_loss.backward()
        optimizer.step()
    return accuracy_score(test_y.numpy(), prediction_classes)

if __name__ == '__main__':
    acc = main()
    assert(acc == 1.0)