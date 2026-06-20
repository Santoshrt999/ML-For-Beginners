import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)


pos = y_train==1
neg=y_train==0

fig,ax=plt.subplots(1,1,figsize=(12,3))
ax.scatter(x_train[pos], y_train[pos], marker='x', c='r', label='y=1 Malignant')
ax.scatter(x_train[neg], y_train[neg], marker='o', c='b', label='y=0 Bengin')
ax.set_title("SCATTER THE DATA")
ax.set_xlabel("tumor size")
ax.set_ylabel("y")
ax.legend()
plt.show()

# 𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖))={
# −log(𝑓𝐰,𝑏(𝐱(𝑖))) if 𝑦(𝑖)=1
# −log(1−𝑓𝐰,𝑏(𝐱(𝑖)))if 𝑦(𝑖)=0


# The loss function above can be rewritten to be easier to implement.
# 𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖))=(−𝑦(𝑖)log(𝑓𝐰,𝑏(𝐱(𝑖)))−(1−𝑦(𝑖))log(1−𝑓𝐰,𝑏(𝐱(𝑖)))

# This is a rather formidable-looking equation. It is less daunting when you consider 𝑦(𝑖)
#  can have only two values, 0 and 1. One can then consider the equation in two pieces:
# when 𝑦(𝑖)=0
# , the left-hand term is eliminated:
# 𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),0)=(−(0)log(𝑓𝐰,𝑏(𝐱(𝑖)))−(1−0)log(1−𝑓𝐰,𝑏(𝐱(𝑖)))=−log(1−𝑓𝐰,𝑏(𝐱(𝑖)))
# and when 𝑦(𝑖)=1
# , the right-hand term is eliminated:
# 𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),1)=(−(1)log(𝑓𝐰,𝑏(𝐱(𝑖)))−(1−1)log(1−𝑓𝐰,𝑏(𝐱(𝑖)))=−log(𝑓𝐰,𝑏(𝐱(𝑖)))



