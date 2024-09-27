import pandas as pd
import numpy as np
import csv
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
#DEVICE = torch.device('cpu') 
# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels, image_ids):
        self.data = data
        self.labels = labels
        self.image_id = image_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.image_id[idx]

train_data1 = pd.read_csv('/home/aalmansour/source/lidc_slices/splits/MNIST_uniform/model0/full_train_set0.csv') 
validation_data1 = pd.read_csv('/home/aalmansour/source/lidc_slices/splits/MNIST_uniform/model0/full_validation_set0.csv')
test_data1 = pd.read_csv('/home/aalmansour/source/lidc_slices/splits/MNIST_uniform/model0/full_test_set0.csv')
print('train full data len:', len(train_data1))
print('val full data len:', len(validation_data1))
print('test full data len:', len(test_data1))
print('----------------------------------------------')

train_data2 = pd.read_csv('/home/aalmansour/source/lidc_slices/splits/MNIST_uniform/model0/high_train_set0.csv') 
validation_data2 = pd.read_csv('/home/aalmansour/source/lidc_slices/splits/MNIST_uniform/model0/high_validation_set0.csv')
test_data2 = pd.read_csv('/home/aalmansour/source/lidc_slices/splits/MNIST_uniform/model0/high_test_set0.csv')
print('train high data len:', len(train_data2))
print('val high data len:', len(validation_data2))
print('test high data len:', len(test_data2))
print('----------------------------------------------')

train_data3 = pd.read_csv('/home/aalmansour/source/lidc_slices/splits/MNIST_uniform/model0/low_train_set0.csv') 
validation_data3 = pd.read_csv('/home/aalmansour/source/lidc_slices/splits/MNIST_uniform/model0/low_validation_set0.csv')
test_data3 = pd.read_csv('/home/aalmansour/source/lidc_slices/splits/MNIST_uniform/model0/low_test_set0.csv')
print('train low data len:', len(train_data3))
print('val low data len:', len(validation_data3))
print('test low data len:', len(test_data3))
print('----------------------------------------------')

train_data4 = pd.read_csv('/home/aalmansour/source/lidc_slices/splits/MNIST_uniform/model0/no_train_set0.csv')
validation_data4 = pd.read_csv('/home/aalmansour/source/lidc_slices/splits/MNIST_uniform/model0/no_validation_set0.csv')
test_data4 = pd.read_csv('/home/aalmansour/source/lidc_slices/splits/MNIST_uniform/model0/no_test_set0.csv')
print('train no data len:', len(train_data4))
print('val no data len:', len(validation_data4))
print('test no data len:', len(test_data4))
print('----------------------------------------------')

input_data = pd.concat([train_data1['flattened_image'], train_data2['flattened_image'], train_data3['flattened_image'], train_data4['flattened_image']], axis=0, ignore_index=True)
val_data = pd.concat([validation_data1['flattened_image'], validation_data2['flattened_image'], validation_data3['flattened_image'], validation_data4['flattened_image']], axis=0, ignore_index=True)
te_data = pd.concat([test_data1['flattened_image'], test_data2['flattened_image'], test_data3['flattened_image'], test_data4['flattened_image']], axis=0, ignore_index=True)

#input_data = pd.concat([train_data1['flattened_image']], axis=0, ignore_index=True)
#val_data = pd.concat([validation_data1['flattened_image']], axis=0, ignore_index=True)
#te_data = pd.concat([test_data1['flattened_image']], axis=0, ignore_index=True)
input_data = input_data.apply(lambda x: ast.literal_eval(x))
input_data = np.stack(input_data)

val_data = val_data.apply(lambda x: ast.literal_eval(x))
val_data = np.stack(val_data)

te_data = te_data.apply(lambda x: ast.literal_eval(x))
te_data = np.stack(te_data)

print('input_data len:', len(input_data))
#print(type(input_data[0]))
print('val_data len:', len(val_data))
print('te_data len:', len(te_data))
print("Images are ready!")
print('----------------------------------------------')

train_image_ids = pd.concat([train_data1['instance_id'], train_data2['instance_id'], train_data3['instance_id'], train_data4['instance_id']], axis=0, ignore_index=True)
val_image_ids = pd.concat([validation_data1['instance_id'], validation_data2['instance_id'], validation_data3['instance_id'], validation_data4['instance_id']], axis=0, ignore_index=True)
test_image_ids = pd.concat([test_data1['instance_id'], test_data2['instance_id'], test_data3['instance_id'], test_data4['instance_id']], axis=0, ignore_index=True)

#train_image_ids = pd.concat([train_data1['instance_id']], axis=0, ignore_index=True)
#val_image_ids = pd.concat([validation_data1['instance_id']], axis=0, ignore_index=True)
#test_image_ids = pd.concat([test_data1['instance_id']], axis=0, ignore_index=True)

train_image_ids = train_image_ids.values
val_image_ids = val_image_ids.values
test_image_ids = test_image_ids.values
print("Image ids are ready!")
print('----------------------------------------------')

train_labels = pd.concat([train_data1['Label'], train_data2['Label'], train_data3['Label'], train_data4['Label']], axis=0, ignore_index=True)
val_labels = pd.concat([validation_data1['Label'], validation_data2['Label'], validation_data3['Label'], validation_data4['Label']], axis=0, ignore_index=True)
test_labels = pd.concat([test_data1['Label'], test_data2['Label'], test_data3['Label'], test_data4['Label']], axis=0, ignore_index=True)

#train_labels = pd.concat([train_data1['Label']], axis=0, ignore_index=True)
#val_labels = pd.concat([validation_data1['Label']], axis=0, ignore_index=True)
#test_labels = pd.concat([test_data1['Label']], axis=0, ignore_index=True)

train_labels = train_labels.values
val_labels = val_labels.values
test_labels = test_labels.values
print(train_labels)
print("Labels are ready!")
print('----------------------------------------------')

batch_size = 32
num_images = len(input_data)
num_val_image = len(val_data)
num_test_image = len(te_data)

# Convert to 2D PyTorch tensors
input_data = torch.from_numpy(input_data).float()
input_data = input_data.unsqueeze(1).to(DEVICE)
tr_ids = torch.tensor(train_image_ids, dtype=torch.long)
#tr_labels = torch.tensor(train_labels, dtype=torch.long)

val_dataset = torch.from_numpy(val_data).float()
val_dataset = val_dataset.unsqueeze(1).to(DEVICE)
val_ids = torch.tensor(val_image_ids, dtype=torch.long)
#va_labels = torch.tensor(val_labels, dtype=torch.long)

test_dataset = torch.from_numpy(te_data).float()
test_dataset = test_dataset.unsqueeze(1).to(DEVICE)
test_ids = torch.tensor(test_image_ids, dtype=torch.long)
#te_labels = torch.tensor(test_labels, dtype=torch.long)
print("Done from tranformation to tensors.")
print('----------------------------------------------')
# Create data loaders
train_loader = DataLoader(CustomDataset(input_data, train_labels, tr_ids), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CustomDataset(val_dataset, val_labels, val_ids), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(CustomDataset(test_dataset, test_labels, test_ids), batch_size=batch_size, shuffle=True)
print("Done from data loaders.")
print('----------------------------------------------')
for images, labels, img_ids in train_loader:
    print('Image batch dimentions:', images.shape)
    print('Image label dimentions:', labels.shape)
    print('Image ids dimentions:', img_ids.shape)
    print('Class labels of 10 examples:', labels[:10])
    break

class CNNModel(nn.Module):
    def __init__(self, num_classes, dropout_prob):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128) # 64 channels, 28x28 feature map size = 50176
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
            x = torch.relu(self.conv1(x))  # Apply convolution and ReLU activation
            x = torch.relu(self.conv2(x))  # Apply another convolution and ReLU activation
            x = x.view(x.size(0), -1)  # Flatten the feature map
            x = torch.relu(self.fc1(x))  # Apply a fully connected layer and ReLU activation
            x = self.dropout(x)  # Apply dropout during training and inferencet
            logits = self.fc2(x)  # Apply the final fully connected layer
            probabilities = torch.softmax(logits, dim=1)  # Apply softmax to get probabilities
            return logits, probabilities
            #return x #torch.softmax(x, dim=1)

model = CNNModel(num_classes=10, dropout_prob=0.3)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 50
epoch_train_accuracies = [] # Create an empty list to store accuracy values
epoch_val_accuracies = []
epoch_test_accuracies = []

# Create empty lists to store predicted labels and image IDs
train_predicted_labels = []
train_image_ids = []
train_probability = []
val_predicted_labels = []
val_image_ids = []
val_probability = []
test_predicted_labels = []
test_image_ids = []
test_probability = []

for epoch in range(num_epochs):
    model.train()
    total_correct = 0
    total_samples = 0
    for images, labels, image_ids in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        image_ids = image_ids
        optimizer.zero_grad()
        images = images.view(32, 1, 28, 28)
        #images = images.view(images.size(0), -1)
        #outputs = model(images)
        logits, probabilities = model(images)
        # Save probabilities
        #torch.save(probabilities, '/home/aalmansour/source/lidc_slices/MNIST/four_raters_probability_labels/baseline_model/training_probabilities.pt')
        loss = criterion(logits, labels)
        loss.backward()
        #total_loss += loss.item()
        optimizer.step()
        # Get predicted train labels
        _, predicted = torch.max(logits, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        
        train_probability.extend(probabilities.tolist())

        # Append predicted labels and image IDs to the lists
        train_predicted_labels.extend(predicted.cpu().numpy())  # Convert to CPU and numpy for compatibility
        train_image_ids.extend(image_ids.cpu().numpy())
    # save probabilities for each epoch
    #torch.save(probabilities, '/home/aalmansour/source/lidc_slices/MNIST/four_raters_probability_labels/MC_dropout30/training_probabilities_epoch{}.pt'.format(epoch))
    train_accuracy = total_correct / total_samples
    epoch_train_accuracies.append(train_accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}] | Training Accuracy: {train_accuracy*100:.2f}%")

    # Validation loop
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for images, labels, image_ids in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            image_ids = image_ids
            images = images.view(32, 1, 28, 28)
            #images = images.view(images.size(0), -1)
            logits, probabilities = model(images)
            _, predicted = torch.max(logits, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            val_probability.extend(probabilities.tolist())

            # Append predicted labels and image IDs to the lists
            val_predicted_labels.extend(predicted.cpu().numpy())  # Convert to CPU and numpy for compatibility
            val_image_ids.extend(image_ids.cpu().numpy())

        val_accuracy = total_correct / total_samples
        epoch_val_accuracies.append(val_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Validation Accuracy: {val_accuracy*100:.2f}%")

        # save probabilities for each epoch
        #torch.save(probabilities, '/home/aalmansour/source/lidc_slices/MNIST/four_raters_probability_labels/MC_dropout30/validation_probabilities_epoch{}.pt'.format(epoch))
    
# Save training predictions and image IDs to a CSV file
with open('train_predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['instance_id', 'Predicted Label', 'probability'])
    writer.writerows(zip(train_image_ids, train_predicted_labels, train_probability))

# Save validation predictions and image IDs to a CSV file
with open('val_predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['instance_id', 'Predicted Label', 'probability'])
    writer.writerows(zip(val_image_ids, val_predicted_labels, val_probability))

# Save the list of accuracies to a file (e.g., CSV or text file)
with open('train_accuracies.csv', 'w') as file:
    # Write the header line
    file.write("Epoch,Accuracy\n")
    # Write the data lines
    for epoch, accuracy in enumerate(epoch_train_accuracies):
        file.write(f"{epoch+1},{accuracy}\n")

with open('val_accuracies.csv', 'w') as file:
        # Write the header line
    file.write("Epoch,Accuracy\n")
    # Write the data lines
    for epoch, accuracy in enumerate(epoch_val_accuracies):
        file.write(f"{epoch+1},{accuracy}\n")

# Test the model
model = model.to(DEVICE)
model.eval()
with torch.no_grad():
    for images, labels, image_ids in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        image_ids = image_ids
        images = images.view(32, 1, 28, 28)
        #images = images.view(images.size(0), -1)
        logits, probabilities = model(images)
        _, predicted = torch.max(logits, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        test_probability.extend(probabilities.tolist())
        # Append predicted labels and image IDs to the lists
        test_predicted_labels.extend(predicted.cpu().numpy())  # Convert to CPU and numpy for compatibility
        test_image_ids.extend(image_ids.cpu().numpy())

    accuracy = total_correct / total_samples
    epoch_test_accuracies.append(accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}] | Testing Accuracy: {accuracy*100:.2f}%")
    # save probabilities for each epoch
    #torch.save(probabilities, '/home/aalmansour/source/lidc_slices/MNIST/four_raters_probability_labels/MC_dropout30/testing_probabilities.pt')
    
# Save testing predictions and image IDs to a CSV file
with open('test_predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['instance_id', 'Predicted Label', 'probability'])
    writer.writerows(zip(test_image_ids, test_predicted_labels, test_probability))

with open('test_accuracies.csv', 'w') as file:
    file.write("Epoch,Accuracy\n")
    for epoch, accuracy in enumerate(epoch_test_accuracies):
         file.write(f"{epoch+1},{accuracy}\n")

''' 
# Define the number of Monte Carlo samples
num_samples = 100

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

# Perform Monte Carlo Dropout inference and calculate uncertainty
def MC_uncertainty(data_loader, model, num_samples, output_csv):
    uncertainty_data = []
    count = 0
    model.eval()
    for images, labels, image_ids in data_loader:
        all_sample_probabilities = []
        all_sample_predictions = []
        all_sample_predictions_std = []
        # Perform Monte Carlo dropout inference for each sample
        for i in range(images.size(0)):
            sample_probabilities = []
            sample_predictions = []
            std_diff_true_pred = []
            images = images.view(32, 1, 28, 28)
            for _ in range(num_samples):
                count +=1
                with torch.no_grad():
                    enable_dropout(model)
                    true_label = image_ids[i].cpu().numpy()
                    current_image = images[i:i+1].to(DEVICE)
                    #output = model(current_image) # Generate Monte Carlo dropout predictions
                    #predicted = torch.argmax(output, dim=1).item()
                    logits, probabilities = model(current_image)
                    _, predicted = torch.max(logits, 1)
                    predicted= predicted.cpu().numpy()
                    probabilities = probabilities.tolist()
                    sample_probabilities.append(probabilities)
                    sample_predictions.append(predicted)
                    std_diff_true_pred.append(true_label-predicted)
            #torch.save(probabilities, '/home/aalmansour/source/lidc_slices/MNIST/four_raters_probability_labels/MC_dropout30/MC_sample_probabilities/MC_probabilities_samples{}.pt'.format(count))
            all_sample_probabilities.append(sample_probabilities)
            all_sample_predictions.append(sample_predictions)
            all_sample_predictions_std.append(np.std(std_diff_true_pred))
        
        # Calculate the mean prediction and variance for each sample
        for i in range(images.size(0)):
            Image_ID = image_ids[i].cpu().numpy()
            true_label = labels[i].item()
            sample_means = sum(all_sample_predictions[i])/len(all_sample_predictions[i])
            sample_std = np.std(all_sample_predictions[i])
            
            uncertainty_data.append([Image_ID, true_label, sample_means, sample_std, true_label-sample_means, all_sample_predictions_std[i], all_sample_predictions[i], all_sample_probabilities[i]])

    # Save the uncertainty data to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['instance_id','True_Label', 'Mean_Prediction', 'Prediction_STD', 'Difference_T-P', 'all_sample_predictions_std', 'all_sample_predictions100', 'all_sample_probabilities'])
        writer.writerows(uncertainty_data)

MC_uncertainty(train_loader, model, num_samples, 'train_uncertainty_stats.csv')
print("MC_uncertainty for training is done!")
MC_uncertainty(val_loader, model, num_samples, 'val_uncertainty_stats.csv')
print("MC_uncertainty for validation is done!")
MC_uncertainty(test_loader, model, num_samples, 'test_uncertainty_stats.csv')
print("MC_uncertainty for testing is done!")
'''