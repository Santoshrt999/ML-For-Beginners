import numpy as np
import matplotlib.pyplot as plt



# The input variable X is a numpy array which has 6 training examples, each with two features
# The output variable y is also a numpy array with 6 examples, and y is either 0 or 1

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

print(f"y shape is {y.shape}")

pos = y==1
neg = y==0

#for 2D array plotting for sigmoid function or logistical regression
fig,ax=plt.subplots(1,1, figsize=(4,4))
ax.scatter(X[pos,0], X[pos,1], marker='x', s=80, c='r', label="y=1")
ax.scatter(X[neg,0], X[neg,1], marker='o', c='b', s=80, label='y=0', edgecolors='blue')
ax.axis((0, 4, 0, 3.5)) #[xmin, xmax, ymin, ymax] use () tuple for axis
ax.set_xlabel("$x^0$")
ax.set_ylabel("$x^1$")
plt.legend()
plt.show()


def sigmoid(z):
    return 1/(1+np.exp(-z))


z=np.arange(-10,11)

fig,ax=plt.subplots(1,1,figsize=(4,4))

ax.plot(z, sigmoid(z), c='b')
ax.set_xlabel("z values")
ax.set_ylabel("sigmoid(z)")
plt.legend()
plt.show()

x0=np.arange(0,6)
x1=3-x0

fig,ax=plt.subplots(1,1,figsize=(4,4))
ax.plot(x0,x1, c='b')
ax.axis((0,4,0,3.5))
ax.fill_between(x0,x1, alpha=0.2)
ax.scatter(X[pos,0], X[pos,1], marker='x', s=80, c='r', label="y=1")
ax.scatter(X[neg,0], X[neg,1], marker='o', c='b', s=80, label='y=0', edgecolors='blue')
ax.axis((0, 4, 0, 3.5)) #[xmin, xmax, ymin, ymax] use () tuple for axis
ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")
plt.show()



