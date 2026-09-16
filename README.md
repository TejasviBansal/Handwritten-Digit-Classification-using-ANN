# Handwritten Digit Classification using ANN

A deep learning project that classifies handwritten digits (0–9) from the standard **MNIST** dataset using an **Artificial Neural Network (ANN)** built with **TensorFlow / Keras**.

---

## 📌 Project Overview

Handwritten digit recognition is a foundational benchmark in computer vision and deep learning. This project demonstrates end-to-end implementation of an Artificial Neural Network (ANN) to classify grayscale digit images (28×28 pixels) into their corresponding numerical values (0 to 9).

The project covers:
- Loading and exploring the MNIST dataset.
- Preprocessing and normalization of pixel values.
- Constructing a Multilayer Perceptron (ANN) architecture.
- Training and validating the model with optimization techniques.
- Model evaluation, performance plotting (loss/accuracy curves), and test predictions.

---

## 🗂️ Dataset: MNIST

The **MNIST (Modified National Institute of Standards and Technology)** dataset contains 70,000 grayscale images of handwritten digits:
- **Training Set:** 60,000 images (28×28 pixels)
- **Testing Set:** 10,000 images (28×28 pixels)
- **Classes:** 10 digits (`0` through `9`)
- **Pixel Intensity:** 0 to 255 (grayscale)

---

## 🛠️ Tech Stack & Dependencies

- **Python 3.8+**
- **TensorFlow / Keras:** Deep learning model building and training
- **NumPy:** Numerical computations and array manipulations
- **Matplotlib / Seaborn:** Data visualization and performance plots
- **Scikit-learn:** (Optional) Metrics evaluation and confusion matrix

Install the dependencies:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn jupyter
```

---

## 🧠 Model Architecture

The Artificial Neural Network (ANN) consists of:

1. **Input / Flatten Layer:**
   - Flattens 2D input images of shape `(28, 28)` into a 1D feature vector of length `784`.
2. **Hidden Layer(s):**
   - Fully Connected (`Dense`) layer with **ReLU** (Rectified Linear Unit) activation to learn non-linear representations.
3. **Output Layer:**
   - Fully Connected (`Dense`) layer with **10 units** and **Softmax** activation, outputting class probability distributions across digits `0`–`9`.

### Summary of Hyperparameters:
| Hyperparameter | Value / Selection |
| :--- | :--- |
| **Loss Function** | `sparse_categorical_crossentropy` / `categorical_crossentropy` |
| **Optimizer** | `Adam` |
| **Metrics** | `accuracy` |
| **Batch Size** | 32 / 64 |
| **Epochs** | 10 – 25 |

---

## 🚀 Workflow & Key Steps

1. **Data Loading:**
   ```python
   import tensorflow as tf
   from tensorflow.keras.datasets import mnist

   (X_train, y_train), (X_test, y_test) = mnist.load_data()
   ```

2. **Data Normalization:**
   Scaling pixel values from `[0, 255]` to `[0, 1]` for faster gradient descent convergence:
   ```python
   X_train = X_train / 255.0
   X_test = X_test / 255.0
   ```

3. **Building the ANN:**
   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Flatten, Dense

   model = Sequential([
       Flatten(input_shape=(28, 28)),
       Dense(128, activation='relu'),
       Dense(32, activation='relu'),
       Dense(10, activation='softmax')
   ])
   model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
   ```

4. **Training the Model:**
   ```python
   history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
   ```

5. **Evaluation & Visualization:**
   - Evaluate test accuracy and loss.
   - Plot training vs. validation accuracy and loss over epochs.
   - Run sample image predictions and inspect classification results.

---

## 📊 Results & Performance

- **Test Accuracy:** Achieves ~**97–98%** test accuracy on unseen MNIST test samples.
- **Inference:** Fast prediction time using `model.predict()` and `np.argmax()` to output predicted digit labels.

---

## 💻 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/TejasviBansal/Handwritten-Digit-Classification-using-ANN.git
   cd Handwritten-Digit-Classification-using-ANN
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook DigitClassification.ipynb
   ```

3. Run all cells sequentially to view data visualizations, model training logs, and prediction benchmarks.

---
