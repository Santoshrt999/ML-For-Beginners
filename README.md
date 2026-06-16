# 🤖 Machine Learning For Beginners

A hands-on, from-scratch journey through the **foundations of machine learning** —
built one concept at a time, in plain NumPy and Matplotlib, with no black-box
libraries hiding the math. This repo follows the path from a straight line
through to classification, implementing each idea by hand to *really* understand
what's going on under the hood.

> The philosophy here: **build it yourself first, then trust the library.**
> Every model is coded from scratch before `scikit-learn` is ever introduced.

---

## 🗺️ The Learning Journey

The project grows in three stages, each building on the last:

```
Linear Regression  ──►  Multiple Linear Regression  ──►  Classification
   (one feature)            (many features)               (0 or 1 labels)
```

---

## 📈 1. Linear Regression — [`linearRegression/`](linearRegression/)

Predicting a continuous number (like house price) from a single input.

| File | What it teaches |
| ---- | --------------- |
| [`machine_learning.py`](linearRegression/machine_learning.py) | The model `f(x) = w·x + b`, training data, predictions |
| [`cost_function.py`](linearRegression/cost_function.py) | Measuring error with the **squared-error cost** |
| [`costfunction_visualisation.py`](linearRegression/costfunction_visualisation.py) | Seeing the cost as a curve we want to minimize |
| [`gradient_descent.py`](linearRegression/gradient_descent.py) | **Learning** `w` and `b` automatically by descending the cost |

**The model — a straight line with ML naming:**

```
f(x) = w·x + b
```

| Symbol | Meaning            |
| ------ | ------------------ |
| `x`    | input feature      |
| `w`    | weight / slope     |
| `b`    | bias / intercept   |
| `f(x)` | prediction         |

> 💡 **Key insight:** "training" just means using **gradient descent** to find
> the `w` and `b` that make the cost as small as possible.

---

## 📊 2. Multiple Linear Regression — [`multiple-linearRegression/`](multiple-linearRegression/)

Same idea, but now each example has **many features**, so the math goes vectorized.

| File | What it teaches |
| ---- | --------------- |
| [`multiple_linearRegression.py`](multiple-linearRegression/multiple_linearRegression.py) | Extending to `f(x) = w·x + b` with vectors |
| [`vectorization.py`](multiple-linearRegression/vectorization.py) | Using `np.dot` instead of slow Python loops |
| [`features_muliregression.py`](multiple-linearRegression/features_muliregression.py) | **Z-score normalization** so features share a scale |
| [`polynomial_features.py`](multiple-linearRegression/polynomial_features.py) | **Feature engineering** — fitting curves with `x², x³` |
| [`scikit_learn.py`](multiple-linearRegression/scikit_learn.py) | Finally, the library version for comparison |

**The dot product is the heart of it:**

```python
z = np.dot(X, w) + b     # all examples, all features, at once
```

> 💡 **Key insights:**
> - **Number of weights = number of features** (not examples!)
> - A *linear* model can fit a *curve* if you feed it engineered features like `x²`
> - Normalizing features makes gradient descent converge much faster

---

## 🎯 3. Classification — [`classification/`](classification/)

When the answer isn't a number but a **category** — yes/no, benign/malignant,
`0` or `1`.

| File | What it teaches |
| ---- | --------------- |
| [`logistical_regression.py`](classification/logistical_regression.py) | Why a straight line **fails** on categorical data |
| [`sigmoid_function.py`](classification/sigmoid_function.py) | The **sigmoid** that squashes any number into `(0, 1)` |
| [`boundaryline.py`](classification/boundaryline.py) | Drawing the **decision boundary** that separates classes |

**The sigmoid function — the star of classification:**

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

It turns the linear output `z = w·x + b` into a **probability**:

```
        z  →  -∞ ........ 0 ........ +∞
  sigmoid →   0 ......... 0.5 ........ 1
            (class 0)  (boundary)  (class 1)
```

> 💡 **Key insights:**
> - The output is a **probability**, not a hard 0/1 — you threshold at **0.5**
> - `sigmoid(z) = 0.5` exactly when `z = 0`, so the **decision boundary** is the
>   line `w·x + b = 0`
> - For `w = [1, 1]`, `b = -3`, the boundary is the clean line `x₁ = 3 - x₀`

---

## 🧠 Concepts Collected Along The Way

A running list of the "aha" moments this project builds:

- **`f(x) = w·x + b`** is the backbone of *both* regression and classification
- **Cost functions** measure how wrong the model is
- **Gradient descent** is how the model learns — nudging `w` and `b` downhill
- **Vectorization** (`np.dot`) replaces slow loops and scales to many features
- **Feature engineering** lets a linear model bend into curves
- **Normalization** keeps features on a comparable scale
- **The sigmoid** converts scores into probabilities for classification
- **Decision boundaries** are just the line where the model flips its answer
- Scientific notation: `e-05` ≈ "basically 0", `9.99e-01` ≈ "basically 1"

---

## 🚀 Running the Code

Each file is standalone — run whichever concept you want to explore:

```bash
python3 linearRegression/gradient_descent.py
python3 classification/sigmoid_function.py
```

Most scripts print details to the console and then open a Matplotlib plot.

## 🛠️ Requirements

- Python 3.8+
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/) *(only for the comparison file)*

```bash
pip install numpy matplotlib scikit-learn
```

## 📚 Credits

Inspired by Andrew Ng's **Machine Learning Specialization** (DeepLearning.AI),
reimplemented from scratch for deeper understanding.

## 📄 License

Released under the [MIT License](LICENSE).
