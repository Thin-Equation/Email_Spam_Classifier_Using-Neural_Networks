# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
import pandas as pd

# Load the dataset
file_path = 'spambase.data'
# Since the dataset does not include header information, we need to create column names
# The dataset description indicates there are 57 attributes followed by a class label
attribute_names = 'word_freq_make, word_freq_address, word_freq_all, word_freq_3d, word_freq_our, word_freq_over, word_freq_remove, word_freq_internet, word_freq_order, word_freq_mail, word_freq_receive, word_freq_will, word_freq_people, word_freq_report, word_freq_addresses, word_freq_free, word_freq_business, word_freq_email, word_freq_you, word_freq_credit, word_freq_your, word_freq_font, word_freq_000, word_freq_money, word_freq_hp, word_freq_hpl, word_freq_george, word_freq_650, word_freq_lab, word_freq_labs, word_freq_telnet, word_freq_857, word_freq_data, word_freq_415, word_freq_85, word_freq_technology, word_freq_1999, word_freq_parts, word_freq_pm, word_freq_direct, word_freq_cs, word_freq_meeting, word_freq_original, word_freq_project, word_freq_re, word_freq_edu, word_freq_table, word_freq_conference, char_freq_;, char_freq_(, char_freq_[, char_freq_!, char_freq_$, char_freq_#, capital_run_length_average, capital_run_length_longest, capital_run_length_total, class_label'.split(', ')

# Read the dataset
spambase_df = pd.read_csv(file_path, names=attribute_names)

# Display the head of the dataframe
print(spambase_df.head())

# Display the shape of the dataframe
print(spambase_df.shape)
  

# %% [markdown]
# EXPRORATORY DATA ANALYSIS

# %%
# Display a summary of the dataframe
print(spambase_df.describe())

# %%
# Check for any missing values
missing_values = spambase_df.isnull().sum()
print('Missing values in each column:\n', missing_values)

# %%
# Check the balance of the classes
spam_class_distribution = spambase_df['class_label'].value_counts()
print('Spam class distribution:\n', spam_class_distribution)

# %%
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style('whitegrid')

# Plotting the distribution of the spam and non-spam classes
plt.figure(figsize=(6, 4))
sns.countplot(x='class_label', data=spambase_df)
plt.title('Distribution of Spam and Non-Spam Classes')
plt.show()

# %%
corr_matrix = spambase_df.corr()
sns.heatmap(corr_matrix)

# %%
from sklearn.model_selection import train_test_split

# Split the data into features and target variable
X = spambase_df.drop('class_label', axis=1)
y = spambase_df['class_label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Output the shape of the train and test sets
print('Training set shape:', X_train.shape)
print('Testing set shape:', X_test.shape)

# %%
from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both the training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# Building Neural Network 

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# Check if GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Convert the scaled data to torch tensors
X_train_tensor = torch.tensor(X_train_scaled.astype(np.float32)).to(device)
y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).to(device)
X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32)).to(device)
y_test_tensor = torch.tensor(y_test.values.astype(np.float32)).to(device)

# %%
# Create TensorDatasets for the training and testing data
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders for the training and testing data
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# %%
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

# %%
# Initialize the neural network
model = SpamClassifier().to(device)

# Define the loss function and optimizer
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
# Function to calculate the accuracy
def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    return accuracy_score(y_true.cpu(), predicted.cpu())


# %%
# Training the neural network
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target.view(-1, 1))
        loss.backward()
        optimizer.step()

    # Print progress
    print('Epoch ', epoch+1, '/', num_epochs, ': Loss -', loss.item())

# %%
# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        predicted = outputs.ge(.5).view(-1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = correct / total
print('Test Accuracy: ', accuracy)


