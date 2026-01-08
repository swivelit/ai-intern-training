import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("outputs", exist_ok=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

G = Generator().to(device)
G.load_state_dict(torch.load("generator.pth", map_location=device))
G.eval()

z = torch.randn(64, latent_dim).to(device)
gen_imgs = G(z)

save_image(gen_imgs.view(64, 1, 28, 28),
           "outputs/generated_digits.png",
           nrow=8, normalize=True)

print("Generated images saved to outputs/generated_digits.png")
