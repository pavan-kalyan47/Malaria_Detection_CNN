# Malaria Detection Using Neural Networks
### Description
This project aims to detect malaria from microscopic cell images using deep learning techniques. By leveraging Convolutional Neural Networks (CNNs) and tools like TensorFlow and Weights & Biases (W&B), this project demonstrates a streamlined process for data preprocessing, model training, evaluation, and hyperparameter optimization.

### Features
Efficient preprocessing of malaria cell images.
Implementation of a LeNet-based CNN architecture for image classification.
Performance tracking and optimization using W&B sweeps.
Evaluation with metrics such as accuracy, precision, recall, and AUC (Area Under the Curve).

### Dataset
Source: ([Malaria Cell Images Dataset](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets))
Classes:
Parasitized
Uninfected
Total Images: 27,558
Images were resized to 224x224 pixels for input into the CNN model and normalized for efficient training.

### Workflow
1. Data Preprocessing
Images were loaded and preprocessed by resizing to 224x224 pixels.
The dataset was split into training (80%), validation (10%), and test (10%) sets.
Data augmentation techniques were applied to enhance model generalization.
2. Model Architecture
The CNN model is based on LeNet, designed for binary classification:

Input Layer: Processes 224x224 RGB images.
Convolution Layers: Two convolutional layers with ReLU activation.
Pooling Layers: MaxPooling layers after each convolution.
Fully Connected Layers: Dense layers for feature aggregation.
Output Layer: A single neuron with sigmoid activation for binary classification.
3. Training
Optimizer: Adam
Loss Function: Binary Crossentropy
Metrics: Accuracy, Precision, Recall
The model was trained for 20 epochs with early stopping to prevent overfitting.
4. Hyperparameter Tuning
W&B sweeps were used to optimize:
Learning rate
Batch size
Dropout rate
Number of filters in convolutional layers
5. Evaluation
The trained model was evaluated using:
Accuracy
Precision
Recall
AUC

### Results
Training Accuracy: 98%
Validation Accuracy: 96%
AUC: 0.99
Inference Time: ~15ms per image on a GPU.

### Results Visualization
Key performance metrics and visualizations:
Training and validation accuracy/loss plots.
ROC curve for model evaluation.
Sample predictions with ground truth label

### Future Work
Experiment with advanced CNN architectures like ResNet and EfficientNet.
Explore transfer learning for faster training and improved accuracy.
Develop a web-based application for real-time malaria detection.
Incorporate more diverse datasets for enhanced model robustness.
