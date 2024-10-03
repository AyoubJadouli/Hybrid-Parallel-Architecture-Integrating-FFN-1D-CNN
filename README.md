# Hybrid Parallel Architecture for Wildfire Prediction

This repository contains the code for a hybrid parallel architecture integrating Feedforward Neural Networks (FFNs), 1D Convolutional Neural Networks (1D CNNs), and Long Short-Term Memory Networks (LSTMs) for predicting wildfire occurrences in Morocco. The architecture is designed to leverage the strengths of each component: FFNs for broad feature transformation, CNNs for spatial pattern recognition, and LSTMs for capturing temporal dependencies.

## Dataset

The model is trained on the "Morocco Wildfire Predictions: 2010-2022 ML Dataset" available on Kaggle: [https://www.kaggle.com/datasets/ayoubjadouli/morocco-wildfire-predictions-2010-2022-ml-dataset](https://www.kaggle.com/datasets/ayoubjadouli/morocco-wildfire-predictions-2010-2022-ml-dataset)

The dataset incorporates a variety of features including meteorological conditions, soil moisture levels, and vegetation indices, which are critical for predicting wildfire occurrences. This dataset is particularly well-suited for LSTMs and 1D CNNs due to its inclusion of lagged features.

## Model Architecture

The architecture begins with a primary input layer that accepts the entire feature set as a flat vector. This input is processed through an FFN block consisting of two dense layers with 128 and 32 neurons, respectively, each followed by batch normalization and dropout layers to prevent overfitting and improve generalization. This block captures broad, non-sequential patterns within the dataset.

For each identified feature subset, the architecture employs parallel processing paths:

*   **CNN Path:** The feature subset is reshaped to form a sequence and processed by a 1D CNN layer with 16 filters and a kernel size of 3. The convolutional layer captures local spatial patterns within the sequence. This is followed by a Global Average Pooling layer to reduce the spatial dimensions, retaining only the most crucial information.
*   **LSTM Path:** Concurrently, the same reshaped feature subset is fed into an LSTM layer with 16 units. The LSTM captures temporal dependencies across the sequence, learning how past values influence future predictions.

The outputs from all CNN and LSTM paths, along with the FFN block, are concatenated into a single feature vector. This combined vector undergoes further processing through additional dense layers with 64 and 32 neurons, each followed by batch normalization and dropout layers. The final dense layer, using sigmoid activation, outputs the probability of the sample belonging to the positive class, making the architecture suitable for binary classification tasks.

## Implementation

The architecture is implemented using TensorFlow and Keras.

## Results

Initial results from training the hybrid neural network architecture demonstrate promising performance in predicting wildfire occurrences. In the first epoch, the model achieved an accuracy of 83.96% on the training set and 87.19% on the validation set. By the second epoch, the model's accuracy further improved to 87.13% on the training set and 87.56% on the validation set.

## Future Work

*   **Hyperparameter Optimization:** Further optimization of hyperparameters, including the number of layers, neurons, and dropout rates, could enhance model performance.
*   **Multi-Class Classification:** Extending the model to handle multi-class classification tasks, such as predicting different levels of wildfire severity, could provide more granular and actionable insights for disaster management.
*   **Incorporation of Additional Data Sources:** Integrating additional data sources, such as real-time satellite imagery or social media feeds, could further improve prediction accuracy and timeliness.
*   **Real-Time Prediction and Deployment:** Implementing the model in a real-time prediction system could provide immediate benefits for wildfire management.
*   **Explainability and Interpretability:** Developing methods to interpret the model's predictions could help stakeholders understand the factors driving wildfire risks.
*   **Cross-Regional Generalization:** Evaluating the model's performance across different geographical regions with varying climatic conditions could test its generalizability and robustness.

## How to Use

1.  Clone the repository
2.  Install the required packages
3.  Download the dataset from Kaggle
4.  Run the `python_training_script.py` script to train the model

## Repository Structure

*   `HybredFFN-CNN-LSTM_Morcco_Ml_WF_2Ep.h5`: Saved model weights
*   `normalisation scaller`: Saved scaler for data normalization
*   `WF_Hybre_parallel_1DCNN_LSTM_FFN.ipynb`: Jupyter notebook with the model code
*   `python_training_script.py`: Python script for training the model
*   `scaler.joblib`: Saved scaler for data normalization

## Authors

*   Ayoub JADOULI
*   Chaker EL AMRANI

## License

This project is licensed under the MIT License.
