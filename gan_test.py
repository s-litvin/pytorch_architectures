import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Parameters
latent_dim = 5
data_dim = 2
batch_size = 64
epochs = 5000
lr = 0.0002
hidden_dim = 64

# Models
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available:", device)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

def get_real_data(batch_size):
    return torch.randn(batch_size, data_dim) * 0.5 + 1


losses_D = []
losses_G = []

for epoch in range(epochs):
    for _ in range(2):
        D.zero_grad()

        real_data = get_real_data(batch_size).to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        output_real = D(real_data)
        loss_real = criterion(output_real, real_labels)

        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_data = G(noise)
        output_fake = D(fake_data.detach())
        loss_fake = criterion(output_fake, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

    G.zero_grad()
    noise = torch.randn(batch_size, latent_dim).to(device)
    fake_data = G(noise)
    output_fake = D(fake_data)
    loss_G = criterion(output_fake, real_labels)
    loss_G.backward()
    optimizer_G.step()

    losses_D.append(loss_D.item())
    losses_G.append(loss_G.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss D={loss_D.item():.4f}, Loss G={loss_G.item():.4f}")


real_samples = get_real_data(100).to(device).detach().cpu().numpy()
noise = torch.randn(100, latent_dim).to(device)
generated_samples = G(noise).detach().cpu().numpy()
print("Data:")
print(generated_samples)

fig, axs = plt.subplots(2, 1, figsize=(8, 10))

axs[0].plot(losses_D, label='Discriminator Loss')
axs[0].plot(losses_G, label='Generator Loss')
axs[0].set_title('GAN Losses')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].grid(True)

axs[1].scatter(real_samples[:,0], real_samples[:,1], color='blue', alpha=0.5, label='Real data')
axs[1].scatter(generated_samples[:,0], generated_samples[:,1], color='red', alpha=0.5, label='Generated data')
axs[1].set_title('Real vs Generated Data')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig('loss_and_data.png')
plt.close()

print("saved to 'loss_and_data.png'")



