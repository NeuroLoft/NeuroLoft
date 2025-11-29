import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from app.model import NeuroCNN
import os

def train():
    # 1. Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Training on: {device}")

    # 2. Data (Auto-download MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("⬇️ Downloading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 3. Model
    model = NeuroCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    model.train()
    epochs = 3
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    # 5. Save Artifact
    if not os.path.exists("app/artifacts"):
        os.makedirs("app/artifacts")
        
    save_path = "app/artifacts/neuro_mnist.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    train()
