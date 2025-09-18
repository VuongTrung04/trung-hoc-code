import torch 
import numpy as np 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

#Chuẩn bị dữ liệu từ file txt
class data_kresling(Dataset):
    def __init__(self, file):
        data = np.loadtxt(file)     # đọc toàn bộ dữ liệu thành mảng numpy 2D 
        # tách dữ liệu
        self.x = torch.tensor(data[:,0:5],dtype=torch.float32)  # feature 
        self.c = torch.tensor(data[:,5:],dtype=torch.float32)   # conditional 
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.c[index]

dataset = data_kresling("kresling_data_CVAE.txt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model CVAE
input_dim = 5   # số input (hình học)
cond_dim = 2    # số condition (frequency, absorption)
latent_dim = 3  # latent space

class CVAE(nn.Module):
    def __init__(self, input_dim,cond_dim,latent_dim):
        super().__init__()
        # encoder 
        self.fc1 = nn.Linear(input_dim + cond_dim,16)
        self.fc_mu = nn.Linear(16,latent_dim)
        self.fc_logvar = nn.Linear(16,latent_dim)

        # decoder 
        self.fc2 = nn.Linear(latent_dim+cond_dim,16)
        self.fc3 = nn.Linear(16,input_dim)

    def encoder(self,x,c):
        xc = torch.cat([x,c],dim=1)  
        h = torch.relu(self.fc1(xc))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)   # dùng Gaussian noise
        return mu + eps*std
    
    def decoder(self,z,c):
        zc = torch.cat([z,c],dim =1)
        h = torch.relu(self.fc2(zc))
        return self.fc3(h)
    
    def forward(self,x,c):
        mu, logvar = self.encoder(x,c)
        z = self.reparameterize(mu,logvar)
        x_recon = self.decoder(z,c)
        return  x_recon, mu, logvar

# hàm mất mát 
def VAE_loss(x_recon,x,mu,logvar):
    recon_loss = nn.functional.mse_loss(x_recon,x,reduction='mean')
    kld = -0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
    return recon_loss+kld

# train CVAE 
model = CVAE(input_dim,cond_dim,latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 20

loss_history = []
for epoch in range(epochs):
    epoch_loss = 0
    for x_batch, c_batch in dataloader:
        optimizer.zero_grad()
        x_recon,mu,logvar = model(x_batch,c_batch)
        loss = VAE_loss(x_recon,x_batch,mu,logvar)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    avg_loss = epoch_loss/len(dataset)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# vẽ loss
plt.plot(loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CVAE Training Loss")
plt.show()

#freeze model
for p in model.parameters():
    p.requires_grad = False
model.eval()

# update trực tiếp vào input
# input ban đầu (random)
x_var = torch.randn((1, input_dim), requires_grad=True)
c_fix = torch.tensor([[5.2, 0.9]], dtype=torch.float32)  # condition cố định

# Target mong muốn (ví dụ)
x_target = torch.tensor([[2.0, 1.0, 0.5, 0.3, 0.1]], dtype=torch.float32)

# Optimizer update trực tiếp input
optimizer_input = optim.Adam([x_var], lr=0.05)

print(" Gradient update on input")
for step in range(50):
    optimizer_input.zero_grad()
    mu, logvar = model.encoder(x_var, c_fix)
    z = model.reparameterize(mu, logvar)
    x_recon = model.decoder(z, c_fix)

    # Mục tiêu: input tạo ra output gần x_target
    loss = nn.functional.mse_loss(x_recon, x_target)
    loss.backward()
    optimizer_input.step()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}, x_var: {x_var.data.numpy()}")

print("\nFinal optimized input:", x_var.data.numpy())
