import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0,2.0])
y_train = np.array([100,300])

print(f"x_train: {x_train}")
print(f"y_train: {y_train}")

#no. of training examples
print(f"Number of training examples: {x_train.shape}")
m=x_train.shape[0]
print(f"Number of training examples: {m}")

m=len(x_train)
print(f"Number of training examples: {m}")


#to Find the ith training example
i=0
i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

#for loop to print all training examples
for i in range(m):
    x_i = x_train[i]
    y_i = y_train[i]
    print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")


# The model function for linear regression (which is a function that maps from x to y) is represented as
# 𝑓𝑤,𝑏(𝑥(𝑖))=𝑤𝑥(𝑖)+𝑏(1)
# The formula above is how you can represent straight lines - different values of  𝑤
#   and  𝑏
#   give you different straight lines on the plot.
# Let's try to get a better intuition for this through the code blocks below. Let's start with  𝑤=100
#   and  𝑏=100
#  .

w=200
b=-100
print(f"w: {w}")
print(f"b: {b}")

# Let's compute the predicted value of y for the first training example (x^0, y^0) using the formula above.

#Example 1:
x_i = x_train[0]
y_i = y_train[0]
y_hat = w*x_i + b
print(f"Predicted value of y for (x^0, y^0): {y_hat}")
y_hat=w*x_train[0] + b
print(f"Predicted value of y for (x^0, y^0): {y_hat}")

#y_hat is the predicted value of y for the first training example (x^0, y^0) using the formula above.
#x= size of the house in square feet
#y= price of the house in dollars
def calculate_y_hat(x, w, b):
    m=x.shape[0]
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i]=w*x[i] + b
    return y_hat

# Plot the data points
y_hat = calculate_y_hat(x_train, w, b)

#Plot should contain the predicted value of y for the first training example (x^0, y^0) using the formula above and the actual value of y for the first training example (x^0, y^0) using a scatter plot. The x-axis should represent the size of the house in square feet and the y-axis should represent the price of the house in dollars.
plt.plot(x_train, y_hat, c='b', label='Predicted Value')

#Scatter should contain the actual value of y for the first training example (x^0, y^0) using a scatter plot. The x-axis should represent the size of the house in square feet and the y-axis should represent the price of the house in dollars.
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Value')
plt.xlabel('House in sq feet')
plt.ylabel('Price in dollars')
plt.title('House Price vs Size') 
plt.legend()#Description of the plot
#plt.show()

print(f"""From the plot above, we can see that the predicted value of y for the first training example (x^0, y^0) using the formula above is not correct. The predicted value of y is 100, while the actual value of y is 300. This means that our model is not able to capture the relationship between x and y correctly. Let's try to find the correct values of w and b that will give us the correct predicted value of y for the first training example (x^0, y^0) using the formula above.""")

print(f"""Predicted value is incorrect.
       Let's try to find the correct values of w and b that will give us the 
      correct predicted value of y for the first training example (x^0, y^0) using the formula above.""")
#f(x) = wx + b
#wx + b = 100
#w*1.0 + b = 100
#w*2.0 + b = 300
#(w*2.0 + b) - (w*1.0 + b) = 300 - 100
#w*2.0 - w*1.0 = 200
#w(2.0 - 1.0) = 200
#w = 200

#Now Plugging w=200 into the first equation to find b
#w*1.0 + b = 100
#200*1.0 + b = 100
#b = 100 - 200
#b = -100

#Correct Values of w and b are w=200 and b=-100
w=200
b=-100
print(f"w: {w}")
print(f"b: {b}")

y_hat = calculate_y_hat(x_train, w, b)

#Plot should contain the predicted value of y for the first training example (x^0, y^0) using the formula above and the actual value of y for the first training example (x^0, y^0) using a scatter plot. The x-axis should represent the size of the house in square feet and the y-axis should represent the price of the house in dollars.
plt.plot(x_train, y_hat, c='b', label='Predicted Value')

#Scatter should contain the actual value of y for the first training example (x^0, y^0) using a scatter plot. The x-axis should represent the size of the house in square feet and the y-axis should represent the price of the house in dollars.
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Value')
plt.xlabel('House in sq feet')
plt.ylabel('Price in dollars')
plt.title('House Price vs Size') 
plt.legend()#Description of the plot
plt.show()


#Let's make original prediction now
x_i=1.5
predictedHouseCost_1500Sqft = w*x_i + b
print(f"Predicted cost of house with 1500 sq ft: ${predictedHouseCost_1500Sqft:.0f}K")