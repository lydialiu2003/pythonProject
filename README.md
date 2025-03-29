# House Price Prediction Using Cascade Correlation Neural Network

This project implements a house price prediction model using the Cascade Correlation neural network architecture. The model is trained on the Boston Housing dataset and demonstrates the effectiveness of Cascade Correlation in predicting housing prices with high accuracy.

## Abstract

The project explores the use of Cascade Correlation, a neural network architecture that automates the learning of network design, to build an accurate and efficient housing price prediction model. By cascading layers incrementally, the model improves its predictions and learns complex patterns in the data. The results show that Cascade Correlation outperforms traditional methods like linear regression in terms of accuracy and convergence speed.

## Problem Statement

Accurate housing price prediction models are crucial for real estate and financial decision-making. This project addresses the question: Can Cascade Correlation neural networks enhance the accuracy of housing price prediction models compared to traditional approaches?

## About Cascade Correlation

Cascade Correlation is a supervised learning algorithm introduced by Scott Fahlman and Christian Lebiere in 1990. It incrementally adds hidden layers to a neural network, optimizing the architecture dynamically. This approach eliminates common issues like vanishing gradients and static network design, making it suitable for complex regression tasks like housing price prediction.

## Methodology

- **Dataset**: The Boston Housing dataset, which includes 14 attributes such as crime rate, average number of rooms, and property tax rate.
- **Libraries Used**: Python, TensorFlow, Keras, and scikit-learn.
- **Model Design**: A Cascade Correlation neural network is implemented using TensorFlow. The model dynamically adds hidden layers to improve accuracy.
- **Training and Testing**: The first 100 samples are used for training, and the next 30 samples are used for testing. The model's performance is evaluated using Mean Squared Error (MSE).

## Results

- **Accuracy**: The model achieved a Mean Squared Error (MSE) of 3.0879, indicating 96.91% accuracy.
- **Comparison**: The Cascade Correlation model significantly outperformed linear regression models, which typically achieve lower accuracy on the same dataset.

## Future Considerations

- **Data Expansion**: Incorporating more diverse datasets can improve the model's generalization and accuracy.
- **Optimization**: Reducing training time by optimizing the number of epochs and exploring alternative architectures.
- **Generalization**: Testing the model on housing data from other cities to validate its applicability beyond Boston.

## How to Run the Project

1. Clone the repository.
2. Install the required Python libraries:
   ```bash
   pip install numpy tensorflow scikit-learn


This README provides an overview of your project, its methodology, results, and instructions for running the code. Let me know if you'd like to make any adjustments!


   