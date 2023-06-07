import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def derivation_CrossEntropy_omega(x1, x2, y, omega1, omega2, eta):
    d_w1 = np.mean(x1 * (sigmoid(x1*omega1 + omega2) - y))
    d_w2 = np.mean(x2 * (sigmoid(x1*omega1 + omega2) - y))
    return d_w1, d_w2, omega1-eta*d_w1, omega2-eta*d_w2


def cross_entropy_loss_1(x1, y, w1, w2):
    c1 = np.mean(y * (x1*w1 + w2))
    c2 = np.mean(np.log(1 + np.exp(x1*w1 + w2)))
    return -1 * (c1-c2)

X1 = np.array([1.2, 2.8, 3.5, 2.2, 3.1, 2.7])
X2 = np.array([1, 1, 1, 1, 1, 1])
Y = np.array([0, 0, 1, 0, 1, 1])
ome_1 = 4
ome_2 = -10
eta = 1
print("loss = " + str(cross_entropy_loss_1(X1, Y, ome_1, ome_2)))
for train_step in range(3):
    print("----------------train_step = " + str(train_step + 1) + "-----------------")
    d_w1, d_w2, ome_1, ome_2 = derivation_CrossEntropy_omega(X1, X2, Y, ome_1, ome_2,eta)
    print("d_w1 = " + str(d_w1))
    print("d_w2 = " + str(d_w2))
    print("omega_1 = " + str(ome_1))
    print("omega_2 = " + str(ome_2))
    print("loss = " + str(cross_entropy_loss_1(X1, Y, ome_1, ome_2)))





