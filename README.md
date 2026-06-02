# Machine Learning For Beginners

A hands-on, beginner-friendly walkthrough of **linear regression** — the
foundational model in machine learning. The example uses a simple
"house price vs. size" dataset to build intuition for how a model maps an
input `x` to a predicted output `f(x)`.

## What's inside

[`machine_learning.py`](machine_learning.py) covers, step by step:

- Creating training data with NumPy arrays (`x_train`, `y_train`)
- Counting training examples (`.shape` vs. `len()`)
- Accessing individual training examples `(x⁽ⁱ⁾, y⁽ⁱ⁾)`
- The linear model function: **`f(x) = w·x + b`**
- Computing predictions (`y_hat`) for one example and for the whole dataset
- Solving for the correct weight `w` and bias `b` by hand using algebra
- Visualizing actual vs. predicted values with Matplotlib

## The model

Linear regression is just the straight-line equation from algebra, with
ML naming conventions:

```
f(x) = w·x + b
```

| Symbol | Meaning            |
| ------ | ------------------ |
| `x`    | input (house size) |
| `w`    | weight / slope     |
| `b`    | bias / intercept   |
| `f(x)` | predicted price    |

For the dataset in this example — points `(1.0, 100)` and `(2.0, 300)` —
the line that fits perfectly is `w = 200`, `b = -100`.

## Requirements

- Python 3.8+
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

Install the dependencies:

```bash
pip install numpy matplotlib
```

## Running it

```bash
python3 machine_learning.py
```

This prints the training-data details and predictions to the console, then
opens a plot showing the actual data points (red ✕) against the model's
prediction line (blue).

## License

Released under the [MIT License](LICENSE).
