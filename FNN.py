import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset,TensorDataset,random_split

import os 
import torch.nn.functional as F
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "origami_data_6in2out.txt")
data = np.loadtxt(file_path,skiprows=1)
X_data = torch.tensor(data[:,0:6],dtype=torch.float32)
y_data = torch.tensor(data[:,6:8],dtype=torch.float32)

X_mean = X_data.mean(0)
X_std = X_data.std(0)
X_norm = (X_data - X_mean)/(X_std+1e-8)
print("X shape:", X_norm.shape)  # (200, 6)
print("Y shape:", y_data.shape)  # (200, 2)
dataset = TensorDataset(X_norm, y_data)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32)
class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,2)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x= self.fc2(x)
        x = F.relu(x)
        x=self.fc3(x)
        return x
model = FNN()
#train 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = 0.01)
epochs = 1000 
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb,yb in train_loader:
        optimizer.zero_grad()
        y_prev = model(xb)
        loss = criterion(y_prev,yb)
        loss.backward()
        optimizer.step()
        total_loss+=loss
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader.dataset):.4f}")
#Freeze 
for p in model.parameters():
    p.requires_grad = False
model.eval()
#check 
model.eval()
with torch.no_grad():
    test_loss = 0
    for xb, yb in test_loader:
        y_prev = model(xb)
        loss = criterion(y_prev,yb)
        test_loss+=loss
    print("Test Loss:", test_loss / len(test_loader.dataset))

# tạo input khi cho output 
y_futr = torch.tensor([[0.5,-0.2]],dtype=torch.float32)
#khởi tạo input 
x_var = torch.rand(1,6,requires_grad=True)
 
optimizer = optim.Adam([x_var],lr=0.01)
criterion = nn.MSELoss()

#train tìm input phù hợp output 
for i in range(200):
    optimizer.zero_grad()
    y_prev = model(x_var)
    loss = criterion (y_prev,y_futr)
    loss.backward()
    optimizer.step()
    if i % 50 == 0:
        print(f"Step {i}: Loss = {loss.item():.4f}")
# kết quả 
input_result_norm = x_var.detach().numpy()
input_result = input_result_norm * X_std.numpy() + X_mean.numpy()
print("Input (giá trị thật):",input_result)






