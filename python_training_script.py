# %% [markdown]
# # Hybrid Parallel Architecture Integrating FFN, 1D CNN, and LSTM for Predicting Wildfire Occurrences in Morocco
# 

# %%
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import joblib
import re
from colorama import Fore, Style, init
import pandas as pd
from collections import defaultdict
import re


# %% [markdown]
# ## Morocco Wildfire Predictions: 2010-2022 ML Dataset
# 
# Download the dataset from: https://www.kaggle.com/datasets/ayoubjadouli/morocco-wildfire-predictions-2010-2022-ml-dataset
# 
# Cite:
# 
# Ayoub Jadouli, and Chaker El Amrani. (2024). Morocco Wildfire Predictions: 2010-2022 ML Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/8040722

# %%

# Initialize colorama
init(autoreset=True)

# Load the dataset
final_dataset_balanced = pd.read_parquet('../Data/Data/FinalDataSet/Date_final_dataset_balanced_float32.parquet')

# Split data into training and validation sets
wf_df_train = final_dataset_balanced[final_dataset_balanced.acq_date < '2022-01-01']
wf_df_valid = final_dataset_balanced[final_dataset_balanced.acq_date >= '2022-01-01']

# Balance the dataset by taking the same number of positive and negative samples
min_samples = min(wf_df_train['is_fire'].value_counts())
wf_df_train_balanced = wf_df_train.groupby('is_fire').apply(lambda x: x.sample(min_samples)).reset_index(drop=True)

min_samples_valid = min(wf_df_valid['is_fire'].value_counts())
wf_df_valid_balanced = wf_df_valid.groupby('is_fire').apply(lambda x: x.sample(min_samples_valid)).reset_index(drop=True)

wf_df_train_balanced = wf_df_train_balanced.sample(frac=1)
wf_df_valid_balanced = wf_df_valid_balanced.sample(frac=1)

# Removing dates
acq_date_train = wf_df_train_balanced.pop('acq_date')
acq_date_valid = wf_df_valid_balanced.pop('acq_date')


# %% [markdown]
# ## Subsets categorizing

# %%

# Function to categorize lagged features based on column names
def categorize_lagged_features_with_index_range(columns):
    lagged_features = defaultdict(list)
    lag_pattern = re.compile(r"(.+)_lag_(\d+)$")
    for idx, column in enumerate(columns):
        match = lag_pattern.search(column)
        if match:
            feature_type = match.group(1)
            lagged_features[feature_type].append(idx)
    feature_ranges = {key: (min(values), max(values)) for key, values in lagged_features.items()}
    return feature_ranges


# %% [markdown]
# ## The Model Implementation
# ### Hybrid Parallel Architecture Integrating FFN, 1D CNN, and LSTM for Predicting Wildfire Occurrences in Morocco
# Tensorflow  Implementation

# %%

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, GlobalAveragePooling1D, Concatenate, BatchNormalization, Dropout, Reshape

def build_model(feature_ranges, total_features):
    # Main input (batch_size, total_features)
    main_input = Input(shape=(total_features,), name='main_input')
    all_outputs = []

    # FFN Block applied to the whole input
    ffn_output = Dense(128, activation='relu')(main_input)
    ffn_output = BatchNormalization()(ffn_output)
    ffn_output = Dropout(0.3)(ffn_output)
    ffn_output = Dense(32, activation='relu')(ffn_output)
    ffn_output = BatchNormalization()(ffn_output)
    ffn_output = Dropout(0.3)(ffn_output)
    
    #Combine outputs from FFN
    all_outputs.append(ffn_output)

    # Iterate through each feature type and setup their respective CNN and LSTM paths
    for feature_type, (start_idx, end_idx) in feature_ranges.items():
        num_features = end_idx - start_idx + 1

        # Slicing input for each feature subset
        subset_input = tf.slice(main_input, [0, start_idx], [-1, num_features])

        # Reshape subset for 1D CNN (batch_size, num_features, 1)
        cnn_input = Reshape((num_features, 1))(subset_input)
        cnn_output = Conv1D(16, kernel_size=3, activation='relu', padding='same')(cnn_input)
        cnn_output = GlobalAveragePooling1D()(cnn_output)

        # Reshape input for LSTM to consider each feature as a timestep (batch_size, num_features, 1)
        lstm_input = Reshape((num_features, 1))(subset_input)
        lstm_output = LSTM(16)(lstm_input)

        # Combine outputs from CNN and LSTM
        combined_output = Concatenate()([cnn_output, lstm_output])
        all_outputs.append(combined_output)

    # Concatenate all outputs from different feature types including FFN output
    final_concatenated = Concatenate()(all_outputs)

    # Final dense layers
    final_dense = Dense(64, activation='relu')(final_concatenated)
    final_dense = BatchNormalization()(final_dense)
    final_dense = Dropout(0.3)(final_dense)

    final_dense = Dense(32, activation='relu')(final_dense)
    final_dense = BatchNormalization()(final_dense)
    final_dense = Dropout(0.3)(final_dense)
    output_layer = Dense(1, activation='sigmoid')(final_dense)  # Assume binary classification

    # Construct the model
    model = Model(inputs=main_input, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# %%

# Split the validation dataset into features and target
X_train = wf_df_train_balanced.drop(columns=['is_fire'])
y_train = wf_df_train_balanced['is_fire']

# Split the validation dataset into features and target
X_valid = wf_df_valid_balanced.drop(columns=['is_fire'])
y_valid = wf_df_valid_balanced['is_fire']

columns = wf_df_valid_balanced.columns
feature_ranges=categorize_lagged_features_with_index_range(columns)

# Initialize a StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training features and transform both the training and validation features
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Save the scaler for later use
joblib.dump(scaler, "../Models/scaler.joblib")

# Assume total features is the maximum index of your features plus one
total_features = int(X_train.shape[1])


# %%

# Build the model with the feature ranges
model = build_model(feature_ranges, total_features)

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=1, validation_data=(X_valid_scaled, y_valid))

# %%
model.save("HybredFFN-CNN-LSTM_Morcco_Ml_WF_2Ep.h5")


