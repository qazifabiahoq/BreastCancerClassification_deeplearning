

# 🩺 Breast Cancer Classification Using Deep Learning (DenseNet201)

This project aims to build a deep learning model for classifying breast cancer images into **Benign** or **Malignant** categories. The approach leverages **transfer learning** with a pretrained **DenseNet201** architecture, combined with real-time **data augmentation** and performance-boosting techniques such as **learning rate scheduling**.

---

## 📦 Dataset Overview

* **Source**: The dataset was obtained from a **Udemy course on medical imaging and deep learning**.
* **Type**: Histopathological images of breast tissue.
* **Classes**:

  * `Benign`: Non-cancerous tissue
  * `Malignant`: Cancerous tissue
* **Structure**: Images organized into two categories (folders), pre-sorted by label.

---

## 🎯 Project Goal

To develop a **binary image classification model** that can distinguish between benign and malignant breast cancer images using **transfer learning**. This model could benefit:

* Medical researchers and practitioners working in cancer diagnostics
* Students and learners exploring applied deep learning in healthcare
* Organizations seeking foundational tools for medical image analysis

---

## 🧠 Why Deep Learning & Transfer Learning?

Medical image datasets are often **small and complex**, making it hard to train models from scratch. By using **DenseNet201**, pretrained on ImageNet, the model benefits from rich feature extraction without needing a large dataset.

---

## 🛠️ Steps & Methodology

### 1. **Mounting Google Drive**

Mounted Google Drive to load the dataset and save the trained model.

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 2. **Data Loading & Preprocessing**

* The dataset zip file was renamed and extracted to avoid issues with special characters.
* Images were categorized into `benign` and `malignant`, converted into NumPy arrays.
* Labels were generated: `0` for benign and `1` for malignant.
* Data was **shuffled** and **one-hot encoded** into two classes.
* Split into **training (80%)** and **validation (20%)** sets.

---

### 3. **Data Visualization**

Sample images were visualized with their corresponding labels to manually verify data quality.

```python
plt.imshow(x_train[i])
# Title: 'Benign' or 'Malignant' based on label
```

---

### 4. **Data Augmentation**

Used `ImageDataGenerator` to create real-time image variations, making the model more robust:

* Random zoom
* Rotation
* Horizontal/vertical flips

```python
train_generator = ImageDataGenerator(
    zoom_range=2,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True
)
```

---

### 5. **Model Building (Transfer Learning)**

Used **DenseNet201** as a **feature extractor**:

* Removed DenseNet201's original classification head.
* Added:

  * Global average pooling
  * Dropout (0.5)
  * Batch normalization
  * Final dense layer with `softmax` for 2-class output

```python
resnet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
```

Model was compiled with:

* `Adam` optimizer (lr = 1e-4)
* `binary_crossentropy` loss
* `accuracy` as evaluation metric

---

### 6. **Training Strategy**

* Trained over **7 epochs** with real-time data augmentation
* Batch size = 16
* Added `ReduceLROnPlateau` callback to dynamically adjust learning rate if validation accuracy plateaued
* Used `math.ceil()` to ensure all batches are included

```python
history = model.fit(
    train_generator.flow(x_train, y_train, batch_size=16),
    steps_per_epoch=math.ceil(len(x_train)/16),
    validation_data=(x_val, y_val),
    epochs=7,
    callbacks=[ReduceLROnPlateau(...)]
)
```

---

### 7. **Training Results**

* **Training accuracy** exceeded 92%
* **Validation accuracy** peaked around **64%**
* Validation loss did not improve significantly, indicating some **overfitting**, possibly due to small dataset size.



---

### 8. **Performance Visualization**

Loss and accuracy graphs were plotted for both training and validation sets:

```python
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['accuracy', 'val_accuracy']].plot()
```


---

## ✅ Key Libraries Used

* **TensorFlow/Keras**: Model building & training
* **OpenCV & PIL**: Image handling
* **Matplotlib**: Visualization
* **NumPy & Pandas**: Data manipulation
* **Scikit-learn**: Train-test split, metrics

---

## 📌 What Was Learned

* How to use **DenseNet201** for transfer learning on medical image data.
* How to apply **data augmentation** to reduce overfitting.
* How to monitor model training using **learning rate schedulers** and **validation metrics**.
* Importance of preprocessing, visualization, and real-time augmentation in small datasets.

---

## 🤝 Who Can Benefit from This

* **Students** learning CNNs and transfer learning in healthcare
* **Healthcare researchers** prototyping cancer detection models
* **Startups** exploring rapid deep learning solutions with minimal data
* **Educators** creating teaching tools for AI in medicine


---

## 📬 Author
Qazi Fabia Hoq : https://www.linkedin.com/in/qazifabiahoq/ 




