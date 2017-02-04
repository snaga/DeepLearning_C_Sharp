# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000,)
print(x_test.shape) # (10000, 784)
print(t_test.shape) # (10000,)

print x_train[0]
x_train_list = x_train.tolist()
print t_train[0]
t_train_list = t_train.tolist()
print x_test[0]
x_test_list = x_test.tolist()
print t_test[0]
t_test_list = t_test.tolist()

f = open("x_train.csv", 'w')
for i in range(0,len(x_train_list)):
    f.write(",".join([str(x) for x in x_train_list[i]]) + "\n")
f.close()

f = open("t_train.csv", 'w')
for i in range(0,len(t_train_list)):
    f.write(str(t_train_list[i]) + "\n")
f.close()

f = open("x_test.csv", 'w')
for i in range(0,len(x_test_list)):
    f.write(",".join([str(x) for x in x_test_list[i]]) + "\n")
f.close()

f = open("t_test.csv", 'w')
for i in range(0,len(t_test_list)):
    f.write(str(t_test_list[i]) + "\n")
f.close()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

f = open("t_train_onehot.csv", 'w')
for i in range(0,len(t_train_list)):
    f.write(str(t_train_list[i]) + "\n")
f.close()

f = open("t_test_onehot.csv", 'w')
for i in range(0,len(t_test_list)):
    f.write(str(t_test_list[i]) + "\n")
f.close()

sys.exit(0)
