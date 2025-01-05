# Blood Cells Cancer Detection Using Vision Transformer (ViT)

## Project Description

This project focuses on the automated detection of **Acute Lymphoblastic Leukemia (ALL)** using advanced machine learning techniques. Leveraging the **Vision Transformer (ViT)** model, coupled with innovative preprocessing methods like **Entropy Filtering** and **Region-Growing Segmentation**, the project enhances diagnostic accuracy for peripheral blood smear (PBS) images.

The model is trained on the **Blood Cells Cancer (ALL) dataset**, achieving a classification accuracy of **98.1%** with minimal loss, demonstrating its efficacy in distinguishing benign and malignant blood cells. This study highlights the potential of transformer-based architectures in medical imaging and aims to facilitate early leukemia diagnosis, improving patient outcomes.

---
## Dataset Download Link

[Blood Cells Cancer (ALL) Dataset](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class )

## Libraries Required

The following libraries are required for this project:

- **Python 3.10**
- **NumPy**: Array and numerical operations
- **Pandas**: Data manipulation
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **OpenCV**: Image processing
- **scikit-image**: Advanced image processing
- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **tqdm**: Progress bars

---

## Methods Used

### 1. Data Preprocessing
- **Entropy Filtering**:
  - Enhances textural details such as cell membranes and nuclei.
  - Calculates local entropy using a disk-shaped sliding window.
  - Improves feature extraction by emphasizing diagnostically relevant areas.
- **Region-Growing Segmentation**:
  - Isolates critical regions, like cell nuclei, by clustering adjacent pixels of similar intensity.
  - Dynamically selects seed points and applies intensity thresholds.

### 2. Data Augmentation
- Techniques like rotation, flipping, zooming, and shearing are employed using the `ImageDataGenerator` from Keras.
- Balances the dataset and mitigates overfitting.

### 3. Model Architecture
- **Vision Transformer (ViT)**:
  - Utilizes the self-attention mechanism for global feature learning.
  - Input images are divided into patches, embedded, and passed through transformer layers.
  - A custom classification head replaces the default for multi-class classification.
- Optimized with the **Adam optimizer**, **binary cross-entropy loss**, and **accuracy metric**.

### 4. Evaluation Metrics
- Accuracy and loss curves
- Confusion matrix for multi-class classification performance


---

## Implementation Details

### Preprocessing
- **Entropy Filtering**:
  - Implemented using `skimage.filters.rank.entropy`.
  - Parameter: Sliding window radius optimized at **5**.
- **Region-Growing Segmentation**:
  - Implemented using `skimage.segmentation.flood`.
  - Parameter: Intensity threshold optimized at **0.25**.

### Model Training
- **Input Size**: Images resized to 224x224 pixels.
- **Training**: 80% of the dataset; **Validation**: 20%.
- **Epochs**: 10.
- **Batch Size**: 32.

---

## Results

- **Proposed Model (ViT + Preprocessing)**:
  - **Accuracy**: 98.1%
  - **Loss**: 0.253
- **Baseline Comparisons**:
  - CNN: 68.5% accuracy
  - CNN with preprocessing: 85.7% accuracy
  - ViT without preprocessing: 92.1% accuracy
- **Confusion Matrix**:
  - High true positive rates across all classes with minimal misclassifications.

---

## Clone the Repository

```bash
git clone https://github.com/UritiSrikanth/Blood-Cells-Cancer-detection-using-CNN.git
```
---

## Install Dependencies

```bash
pip install -r requirements.txt
```