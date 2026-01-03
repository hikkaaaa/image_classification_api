import torch 
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys 
import os

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#add the parent directory to system path so we can import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model import SimpleCNN
def train(): 
    #setup device (if abailable - gpu, if not cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #prepare dat  (cifar-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Downloading data...")
    train_set = torchvision.datasets.CIFAR10(root='./ml_experiments/data', train = True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Starting training...")
    for epoch in range(3): 
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0): 
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999: 
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished training")
    
    # 6. Save the Model
    # Get the directory where this script (train.py) lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the project root, then into 'models'
    save_dir = os.path.join(script_dir, '..', 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'classifier.pth')
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()