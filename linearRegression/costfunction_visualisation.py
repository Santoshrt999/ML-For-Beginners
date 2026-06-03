import numpy as np
import matplotlib.pyplot as plt
import statistics


x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730])

# Example predicted values (replace with actual predictions from your model)

#PREDICTION for each training example
def predict(x, w, b):
    m = x.shape[0] #m is the number of training examples
    f_x = np.zeros(m)          # one predicted value per training example
    for i in range(m): #i is the index of the training example
        f_x[i] = w*x[i] + b
    return f_x

#COST FUNCTION DEFINITION
def compute_cost(x,y,w,b):
    m=x.shape[0]
    sum_cost=0
    for i in range(m):
        f_x=w*x[i]+b-y[i]   #prediction - actual
        cost=f_x**2
        sum_cost+=cost
    return 1/(2*m)*sum_cost

# Pick a line, then measure how good it is with the cost function
w = 190
b = 45
y_hat = predict(x_train, w, b)              # the line (one value per x)
cost = compute_cost(x_train, y_train, w, b)  # the single "badness" score
print(f"Cost J(w={w}, b={b}) = {cost:.2f}")

plt.plot(x_train, y_hat, c='b', label='Predicted Value')

plt.scatter(x_train, y_train, color='red', marker='x', label='Actual Value')
plt.xlabel('Size of house (x)')
plt.ylabel('Price of house (y)')
plt.title(f'Training data  (w={w}, b={b}, cost={cost:.0f})')
plt.legend()
plt.show()

#Least Sqaure Formula

#Step 1: Calculate the mean of x and y
meanX = statistics.mean(x_train)
meanY = statistics.mean(y_train) #Y HAT = wX + b

#step 2 get deviations

#x-axis contains known data like sq feet
dx_dy=[]
dx_squared=[]
for i in range(len(x_train)):
    x_dev =x_train[i] - meanX
    y_dev=y_train[i] - meanY
    dx_dy.append(x_dev*y_dev)
    dx_squared.append(x_dev**2)


#Step 3: Calculate the slope (w) 
w= sum(dx_dy)/sum(dx_squared)

#step 4: Calculate the intercept (b)
#fx=wx+b is same as meanY = W*meanX + b, so we can rearrange to get b = meanY - w*meanX
b= meanY - w*meanX #Use the mean of x and y to calculate the intercept b
print(f"w: {w}")
print(f"b: {b}")


y_hat = predict(x_train, w, b)              # the line (one value per x)
cost = compute_cost(x_train, y_train, w, b)  # the single "badness" score
print(f"Cost J(w={w}, b={b}) = {cost:.2f}")

plt.plot(x_train, y_hat, c='b', label='Predicted Value')

plt.scatter(x_train, y_train, color='red', marker='x', label='Actual Value')
plt.xlabel('Size of house (x)')
plt.ylabel('Price of house (y)')
plt.title(f'Training data  (w={w}, b={b}, Optimal Cost={cost:.0f})')
plt.legend()
plt.show()

# Mean of x and y.
# Each point: dx = x − x̄, dy = y − ȳ.
# Compute dx·dy and dx².
# w = Σ(dx·dy) ÷ Σ(dx²).
# b = ȳ − w·x̄ (using the means).





