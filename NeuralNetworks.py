import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define the neural network architecture
class SpamClassifier(nn.Module):
    def __init__(self):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(57, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load your dataset
# Assuming 'spambase.data' is a CSV file with the last column as the target
data = np.loadtxt('spambase.data', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert the data to torch tensors
X_train_tensor = torch.tensor(X_train.astype(np.float32))
y_train_tensor = torch.tensor(y_train.astype(np.float32))
X_test_tensor = torch.tensor(X_test.astype(np.float32))
y_test_tensor = torch.tensor(y_test.astype(np.float32))

# Load the model (if you have a pre-trained model, load its weights here)
model = SpamClassifier()

# Streamlit app
def main():
    st.title('Spam Classifier')
    
    # User input features
    # You can create input fields for each feature or use a single text input to accept an email
    # For simplicity, here's a single input for a feature vector
    user_input = st.text_input('Enter your feature vector (comma-separated values):')
    
    if user_input:
        # Process user input
        features = np.array([float(x) for x in user_input.split(',')])
        features = scaler.transform(features.reshape(1, -1))  # Scale the features
        features_tensor = torch.tensor(features.astype(np.float32))
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            prediction = model(features_tensor)
            predicted_class = 'Spam' if prediction.item() > 0.5 else 'Not Spam'
        
        # Display prediction
        st.subheader(f'The email is classified as: {predicted_class}')

if __name__ == '__main__':
    main()