import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL (varying size)
class SimpleCNN(nn.Module):
    def __init__(self, width=32):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width*2, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(width*2, 10)
        )

    def forward(self, x):
        return self.net(x)

# DATASET
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# TRAIN FUNCTION
def train_model(model, trainloader, testloader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_acc, test_acc = [], []

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100. * correct / total
        train_acc.append(train_accuracy)

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_accuracy = 100. * correct / total
        test_acc.append(test_accuracy)

        print(f"Epoch {epoch+1}: Train {train_accuracy:.2f}%, Test {test_accuracy:.2f}%")

    return train_acc, test_acc

# EXPERIMENTS
dataset_sizes = [1000, 5000, 10000]
model_widths = [16, 32, 64]

results = {}

for size in dataset_sizes:
    indices = np.random.choice(len(trainset), size, replace=False)
    subset = Subset(trainset, indices)

    trainloader = DataLoader(subset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    for width in model_widths:
        print(f"\nDataset size: {size}, Model width: {width}")

        model = SimpleCNN(width).to(device)
        train_acc, test_acc = train_model(model, trainloader, testloader)

        results[(size, width)] = (train_acc, test_acc)

# PLOTTING
plt.figure(figsize=(10,6))

for (size, width), (train_acc, test_acc) in results.items():
    plt.plot(test_acc, label=f"size={size}, width={width}")

plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.title("Scaling Behavior of Models")
plt.legend()
plt.savefig("scaling_results.png")

print("Plot saved as scaling_results.png")
