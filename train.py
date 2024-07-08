import torch
import torch.optim as optim
from tqdm import tqdm
import os

def masked_mse_loss(output, target, mask):
    loss = (output - target) ** 2
    loss = loss * mask
    return loss.sum() / mask.sum()

def train_model(model, data_loader, num_epochs=20, learning_rate=0.003):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for features, labels in tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            
            # Create mask where features are -1 and labels are not -1
            mask = (features == -1) & (labels != -1)
            mask = mask.float()
            
            # Setting NaN targets to zero
            labels[labels == -1] = 0

            loss = masked_mse_loss(outputs, labels, mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}')
        return

def save_model(model):
    model_folder = "model"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    # Save the model
    torch.save(model.state_dict(), f"{model_folder}/model.pth")
