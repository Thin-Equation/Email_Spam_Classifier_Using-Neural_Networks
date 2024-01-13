import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'spambase.data'
attribute_names = 'word_freq_make, word_freq_address, word_freq_all, word_freq_3d, word_freq_our, word_freq_over, word_freq_remove, word_freq_internet, word_freq_order, word_freq_mail, word_freq_receive, word_freq_will, word_freq_people, word_freq_report, word_freq_addresses, word_freq_free, word_freq_business, word_freq_email, word_freq_you, word_freq_credit, word_freq_your, word_freq_font, word_freq_000, word_freq_money, word_freq_hp, word_freq_hpl, word_freq_george, word_freq_650, word_freq_lab, word_freq_labs, word_freq_telnet, word_freq_857, word_freq_data, word_freq_415, word_freq_85, word_freq_technology, word_freq_1999, word_freq_parts, word_freq_pm, word_freq_direct, word_freq_cs, word_freq_meeting, word_freq_original, word_freq_project, word_freq_re, word_freq_edu, word_freq_table, word_freq_conference, char_freq_;, char_freq_(, char_freq_[, char_freq_!, char_freq_$, char_freq_#, capital_run_length_average, capital_run_length_longest, capital_run_length_total, class_label'.split(', ')
spambase_df = pd.read_csv(file_path, names=attribute_names)

# Data preprocessing
X = spambase_df.drop('class_label', axis=1)
y = spambase_df['class_label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both the training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the scaled data to torch tensors
X_train_tensor = torch.tensor(X_train_scaled.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.astype(np.float32))

# Create TensorDatasets for the training and testing data
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders for the training and testing data
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define the neural network architecture
class SpamClassifier1(nn.Module):
    def __init__(self, activation_fn):
        super(SpamClassifier1, self).__init__()
        self.fc1 = nn.Linear(57, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.activation_fn = activation_fn()

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
class SpamClassifier2(nn.Module):
    def __init__(self, activation_fn):
        super(SpamClassifier2, self).__init__()
        self.fc1 = nn.Linear(57, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.activation_fn = activation_fn()

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        x = self.activation_fn(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# Streamlit app
def main():
    st.title("Spam Classifier App")

    # Display the dataset
    st.subheader("Dataset Preview")
    st.dataframe(spambase_df.head())

    # Model selection
    classifier_options = {
        "SpamClassifier_ReLU_3Layer": SpamClassifier1(nn.ReLU),
        "SpamClassifier_Tanh_3Layer": SpamClassifier1(nn.Tanh),
        "SpamClassifier_Sigmoid_3Layer": SpamClassifier1(nn.Sigmoid),
        "SpamClassifier_Identity_3Layer": SpamClassifier1(lambda: nn.Identity()),
        "SpamClassifier_ReLU_4Layer": SpamClassifier2(nn.ReLU),
        "SpamClassifier_Tanh_4Layer": SpamClassifier2(nn.Tanh),
        "SpamClassifier_Sigmoid_4Layer": SpamClassifier2(nn.Sigmoid),
        "SpamClassifier_Identity_4Layer": SpamClassifier2(lambda: nn.Identity()),
    }

    selected_classifier = st.selectbox("Select Spam Classifier", list(classifier_options.keys()))

    # Train and evaluate the selected classifier
    model = classifier_options[selected_classifier]
    train_and_evaluate(model)

    # Input features from the user
    st.subheader("Enter Features for Prediction")
    user_input = {}
    for feature_name in X.columns:
        user_input[feature_name] = st.number_input(f"{feature_name}:", value=10.0)

    # Convert user input to torch tensor
    user_input_tensor = torch.tensor([list(user_input.values())], dtype=torch.float32)

    # Make prediction
    prediction = predict_spam(user_input_tensor, model)

    # Display the prediction
    st.subheader("Prediction")
    st.write(f"The predicted probability of being spam: {prediction.item()}")

# Function to train and evaluate the model
def train_and_evaluate(model):
    # Initialize the neural network
    model = model

    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the neural network
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target.view(-1, 1))
            loss.backward()
            optimizer.step()

        # Print progress
        print('Epoch ', epoch+1, '/', num_epochs, ': Loss -', loss.item())

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in test_loader:
            outputs = model(data)
            predicted = outputs.ge(.5).view(-1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total

    # Display accuracy
    st.subheader("Model Evaluation")
    st.write(f"Test Accuracy: {accuracy}")

    # Save the model (optional)
    if st.button("Save Model"):
        torch.save(model.state_dict(), 'spam_classifier_model.pt')
        st.write("Model saved successfully.")

# Function to make predictions
def predict_spam(features, model):
    with torch.no_grad():
        output = model(features)
        return output.squeeze().numpy()

if __name__ == "__main__":
    main()
