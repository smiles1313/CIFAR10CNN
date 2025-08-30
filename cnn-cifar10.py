import torch
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

# Checks if CUDA is availiable, loads the device for computation to the GPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Running on GPU')
    print(torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print('Running on CPU')

# Data Processing
train_dataset = torchvision.datasets.CIFAR10(root='./cifar10', transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, transform=torchvision.transforms.ToTensor(), download=True)

# Batching
train_loader = torch.utils.data.DataLoader(train_dataset, 128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, 128, shuffle=True)

# Visualizing a sample from train loader
print(train_dataset)

# Extracting an image and label from the dataset
train_iter = iter(train_loader)
batch_images, batch_labels = next(train_iter)
image, label = batch_images[0], batch_labels[0]

print(image.shape)
plt.imshow(image.permute(1, 2, 0)) # permute 1st 2nd 0th index is 32 32 3
plt.show()

#Class module to create a CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First convolutional layer
        # Here we're defining a standard layer with Convolution, BatchNorm, and dropout
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2)
                              #og, new dim
        # b x 3 x 32 x 32 -> b x 32 x 16 x 16, image divided by 2
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU() # Using ReLU activation function
        self.dropout1 = nn.Dropout(0.1) # Adding dropout to prevent overfitting (recommend a rate of 0.1)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2) # b x 32 x 16 x 16 -> b x 64 x 8 x 8
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) # Adding a pooling layer to reduce spatial dimensions, b x 64 x 8 x 8 -> b x 64 x 4 x 4
        self.dropout2 = nn.Dropout(0.1) # Recommend rate of 0.05

        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2) # b x 64 x 4 x 4 -> b x 64 x 4 x 4
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.05) # Recommend rate of 0.05
        self.flatten = nn.Flatten()  # b x 64 x 4 x 4 -> b x (64 * 4 * 4)

        # Fully connected layer - classifying the features into 10 classes
        #linear bc you flatten out results of other conv layers for classification
        self.fc = nn.Linear(64 * 4, 128) # 64 from the last conv layer, 10 for the number of classes, b x (64 * 4 * 4) -> b x 128
        self.relu4 = nn.ReLU()
        self.fc1 = nn.Linear(128, 10)  # b x 128 -> b x 10

    # This is already done - we're just calling the functions we define
    def forward(self, x):
        # Describing the forward pass through the network
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # After all those conv layers we can finally pass into a fully connected layer
        print(x.shape)  #for debugging
        x = self.flatten(x)  # Flattening the output of the conv layers for the fully connected layer
        print(x.shape)  #for debugging
        x = self.fc(x)
        x = self.relu4(x)
        x = self.fc1(x)
        return x  # The softmax (or another activation) can be implicitly applied by the loss function

input = torch.randn(32, 3, 32, 32)
model = CNN()
output = model(input)
print(output.shape)

# Creating an instance of our CNN model
model = CNN()
model.to(device)
criterion = nn.CrossEntropyLoss()  # define our loss

# Define the optimizer here, the model.parameters() are all the parameters of our model, lr is the learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)

# Training Loop
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    for i, batch in tqdm(enumerate(train_loader)):  # looping through
        inputs, labels = batch # The batch contains the inputs and labels
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Get the model output
        outputs = model(inputs)
        # Calculate the loss (using the criterion defined above)
        loss = criterion(outputs, labels) #run it thru the criterion defined before as cross entropy loss
        # Call loss.backward - this actually computes the gradients
        loss.backward()
        # Step forward with the optimizer and then zero out the gradients
        optimizer.step()
        optimizer.zero_grad() #zero out gradients
    print('End of epoch loss:', round(loss.item(), 3))
    
# Testing
@torch.no_grad() # Letting torch know we don't need the gradients as we are only testing
def test(model, test_loader, device):
    # Manually specified the classes - these are from the cifar-10 dataset
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # Put the model in evaluation mode
    model.eval()
    correct = 0
    for i, batch in tqdm(enumerate(test_loader)):
         inputs, labels = batch
         inputs = inputs.to(device)
         labels = labels.to(device)
         # Get the output
         outputs = model(inputs)
         # Determine if it made the right prediction
         predictions = outputs.argmax(dim=1) #take max of predictions and max probability
         # If it made the right prediction, increase the number correct
         correct += (predictions == labels).sum().item() #??? what does this mean
    print(f"End of epoch accuracy: {100 * correct / len(test_dataset)}%")

    # visualizing the current model's performance
    for i in range(min(len(inputs), 8)):
        print('Guess:', classes[predictions[i]], '| Label:', classes[labels[i]])
        plt.imshow(inputs[i].cpu().permute(1,2,0))
        plt.show()


# Running the train-test loop
NUM_EPOCHS = 2 # One epoch is one loop through the training data
for epoch in range(NUM_EPOCHS):
    print("Epoch: ", epoch + 1)
    train_one_epoch(model, train_loader, optimizer, criterion, device)
    test(model, test_loader, device)
size = 0
for param in model.parameters():
    size += np.prod(param.shape)
print(f"Number of parameters: {size}")


# Saving the weights
torch.save(model.state_dict(), "model.pth")

# Reloading the weights just saved
model_new = CNN()
model_new.load_state_dict(torch.load("model.pth"))
model_new.to(device)
model_new.eval()

test(model_new, test_loader, device)
