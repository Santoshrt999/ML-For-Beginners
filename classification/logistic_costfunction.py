import numpy as np
import matplotlib.pyplot as plt

#Layout for ML in general

# A 2D training array is always laid out as:
# X.shape = (m, n) = (number of examples, number of features)
#             ↑              ↑
#          shape[0]       shape[1]
#          = examples     = FEATURES
# So:

# X.shape[0] → how many examples (rows)
# X.shape[1] → how many features (columns) ← this is what you want

# X_train.shape        # (6, 2)
# X_train.shape[0]     # 6  → 6 examples
# X_train.shape[1]     # 2  → 2 features   ✅

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1]) 

print(f"X-train shaeps {X_train.shape}")

print(f"X-train shaeps {X_train[0].shape}")
print(f"X-train shaeps {X_train[1].shape}")

fig,ax=plt.subplots(1,1,figsize=(4,4))
pos = y_train == 1
neg = y_train == 0
ax.scatter(X_train[pos,0], X_train[pos, 1], marker='x', c='r', label="y=1")
ax.scatter(X_train[neg,0], X_train[neg, 1], marker='o', c='b', label="y=0")
ax.axis((0, 4, 0, 3.5))
ax.set_title("Plot graphs of examples")
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
plt.legend()
plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖))=−𝑦(𝑖)log(𝑓𝐰,𝑏(𝐱(𝑖)))−(1−𝑦(𝑖))log(1−𝑓𝐰,𝑏(𝐱(𝑖)))
def compute_logistic_cost(X,Y,W,b):

    """
    x,y,w are ndarray and b is scalar
    """

    m=X.shape[0]
    cost=0.0
    for i in range(m):
        z_i = np.dot(X[i],W) +b #fx=x*w+b
        g_z = sigmoid(z_i)
        cost += -Y[i] * np.log(g_z) -(1 - Y[i])* np.log(1-g_z)
    cost = cost/m
    return cost    

W_init=np.array([1,1])
b=-3


print(f"Cost of logist regression for data is {compute_logistic_cost(X_train,y_train,W_init,b)}")

x0=np.arange(0,6)

#Boundary line
x1 = 3 - x0
x1_other = 4 - x0

#b=-3 and b=-4
fig,ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(x0,x1, c='b', label="$b$=-3")
ax.plot(x0,x1_other, c='m', label="$b$=-4")
ax.axis((0, 4, 0, 4))

ax.scatter(X_train[pos,0], X_train[pos, 1], marker='x', c='r', label="y=1")
ax.scatter(X_train[neg,0], X_train[neg, 1], marker='o', c='b', label="y=0")

ax.set_xlabel('$x_0')
ax.set_ylabel('$x_1')
plt.title('Decision Boundary')
plt.show()


# The size of W depends on the number of features — specifically, one weight per feature.
w_array1 = np.zeros(X_train.shape[1])
b_1 = -3
w_array2 = np.array(X_train.shape[1])
b_2 = -4

print("Cost for b = -3 : ", compute_logistic_cost(X_train, y_train, w_array1, b_1))
print("Cost for b = -4 : ", compute_logistic_cost(X_train, y_train, w_array2, b_2))







