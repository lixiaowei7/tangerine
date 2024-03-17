import time
import math
import numpy as np

from multiprocessing import Pool
from datasets import load_dataset
from line_profiler import LineProfiler


PROCESS_NUM = 30


def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


class KNN:

    def __init__(self, k, label_num):
        self.k = k
        self.label_num = label_num

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def get_knn_indices(self, x):
        dis = list(map(lambda a: distance(a, x), self.x_train))
        knn_indices = np.argsort(dis)
        knn_indices = knn_indices[:self.k]
        return knn_indices

    def get_label(self, x):
        knn_indices = self.get_knn_indices(x)

        label_statistic = np.zeros(shape=[self.label_num])
        for index in knn_indices:
            label = int(self.y_train[index])
            label_statistic[label] += 1
        return np.argmax(label_statistic)

    def get_labels(self, xs, s, e):
        return [self.get_label(x) for x in xs], s, e

    def predict(self, x_test):
        predicted_test_labels = np.zeros(shape=[len(x_test)], dtype=int)
        results = []
        pool = Pool(processes=PROCESS_NUM)
        batch = math.ceil(len(x_test) / PROCESS_NUM)
        for i in range(PROCESS_NUM):
            s = i * batch
            e = (i + 1) * batch
            result = pool.apply_async(self.get_labels, (x_test[s:e], s, e))
            results.append(result)

        for r in results:
            labels, s, e = r.get()
            predicted_test_labels[s:e] = labels
        return predicted_test_labels


def main():
    dataset = load_dataset('mnist')
    x_train = np.array(list(map(np.array, dataset['train']['image']))).astype(np.float32)
    y_train = dataset['train']['label']

    x_test = np.array(list(map(np.array, dataset['test']['image'])))
    y_test = dataset['test']['label']

    for k in range(1, 2):
        s = time.time()
        knn = KNN(k, label_num=10)
        knn.fit(x_train, y_train)
        predicted_labels = knn.predict(x_test)

        print(len(predicted_labels))
        accuracy = np.mean(predicted_labels == y_test)
        print(k, accuracy, time.time() - s)


if __name__ == '__main__':
    # 1 0.2727
    # 2 0.2432

    lp = LineProfiler()
    lp.add_function(KNN.predict)
    lp_wrapper = lp(main)
    lp_wrapper()
    lp.print_stats()
