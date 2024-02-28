import torch
import numpy as np

from datasets import load_dataset

device = torch.device('cuda:0')


class KNN:

    def __init__(self, k, label_num):
        self.k = k
        self.label_num = label_num

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def get_label(self, x):
        dis = (self.x_train - x).square().sqrt().sum(1)
        knn_indices = torch.argsort(dis)
        indices = knn_indices[:self.k]

        label_statistic = torch.zeros(self.label_num, device=device)
        for index in indices:
            label = int(self.y_train[index])
            label_statistic[label] += 1
        return torch.argmax(label_statistic)

    def predict(self, x_test):
        predicted_test_labels = torch.zeros(x_test.shape[0], dtype=int, device=device)
        for i, x in enumerate(x_test):
            predicted_test_labels[i] = self.get_label(x)
        return predicted_test_labels


def main():
    dataset = load_dataset('mnist').with_format('torch', device=device)
    x_train = dataset['train']['image']
    x_train_shape = x_train.shape
    x_train = x_train.reshape(x_train_shape[0], x_train_shape[1] * x_train_shape[2])
    y_train = dataset['train']['label']

    x_test = dataset['test']['image']
    x_test_shape = x_test.shape
    x_test = x_test.reshape(x_test_shape[0], x_test_shape[1] * x_test_shape[2])
    y_test = dataset['test']['label']

    for k in range(1, 10):
        knn = KNN(k, label_num=10)
        knn.fit(x_train, y_train)
        predicted_labels = knn.predict(x_test)

        accuracy = torch.mean((predicted_labels == y_test).float())
        print(k, accuracy)


if __name__ == '__main__':
    # 1 0.5217
    # 2 0.4510

    main()
