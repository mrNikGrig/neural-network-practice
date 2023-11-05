import matplotlib.pyplot as plt
import numpy as np
import gc
from math import *
from tqdm import tqdm
import numpy.random

#globals
global data_iris
global error


def mse(arr):
    s = 0
    for i in range(len(arr)):
        s += pow(1-arr[i], 2)
    try:
        return s / len(arr)
    except:
        return 1


def sigmoid(x):
    return 1/(1+exp(-x))


def back_propagation():
    pass


def main():
    data_iris = [[]]
    f = open('iris.txt', 'r').readlines()
    data_iris[0] = list(
        map(float, f[0].split()))  # считываем данные из файла получаем массив массивов[[данные ириса и его вид]]
    for i in range(1, len(f)):
        data_iris.append(list(map(float, f[i].split())))
    del f
    gc.collect()

    # train = []
    # test = []
    # for i in range(len(data_iris)): # разделение исходных данных на две выборки: тренировочную(80%) и тестирующую(20%)
    #     if i % 50 < 50 * 0.8:
    #         train.append(data_iris[i])
    #     else:
    #         test.append(data_iris[i])
    n1 = [0] * 4
    n2 = [0] * 5 # требуется анализ для более эффективного строения перцептрона
    n3 = [0] * 3
    np.random.seed(0)
    accuracy = []
    w1, w2 = [], []

    while mse(accuracy) > 0.2:
        counter_error = 0
        accuracy.clear()
        w1 = np.random.randn(len(n1), len(n2))
        w2 = np.random.randn(len(n2), len(n3))
        for i in range(len(data_iris)):
            n1 = data_iris[i][:-1]
            for it in range(len(n1)):
                for jt in range(len(n2)):
                    n2[jt] += w1[it][jt] * n1[it]

            n2 = list(map(sigmoid, n2))
            for it in range(len(n2)):
                for jt in range(len(n3)):
                    n3[jt] += w2[it][jt] * n2[it]

            n3 = list(map(sigmoid, n3))
            if n3.index(max(n3)) + 1 == data_iris[i][4]:
                accuracy.append(max(n3))
            else:
                counter_error +=1
                # accuracy.append(n3[round(data_iris[i][4] - 1)])
                accuracy.append(0)
        print(counter_error, counter_error/150, mse(accuracy))
    f = open('weight.txt', 'w')
    f.write(str(w1))
    f.write(str(w2))
    f.close()


if __name__ == "__main__":
    main()
    # arr = range(-10,10)
    # new_arr = list(map(sigmoid, arr))
    # plt.plot(arr, new_arr)
    # plt.show()
