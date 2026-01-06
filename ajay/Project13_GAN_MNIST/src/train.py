import os
import argparse
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


@dataclass
class Config:
    # Data
    data_dir: str = "data"
    image_size: int = 28
    batch_size: int = 128
    num_workers: int = 2

    # Model
    latent_dim: int = 100
    g_channels: int = 64
    d_channels: int = 64

    # Train
    epochs: int = 30
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging / outputs
    out_dir: str = "outputs"
    save_every: int = 1
    fixed_grid_n: int = 64  # 8x8 grid


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Generator(nn.Module):
    """
    DCGAN-ish generator for 28x28x1.
    Start from latent vector -> (g_channels*4) x 7 x 7 -> upsample to 14 -> 28.
    """
    def __init__(self, z_dim=100, g_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, g_channels * 4, 7, 1, 0, bias=False),  # 7x7
            nn.BatchNorm2d(g_channels * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_channels * 4, g_channels * 2, 4, 2, 1, bias=False),  # 14x14
            nn.BatchNorm2d(g_channels * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_channels * 2, g_channels, 4, 2, 1, bias=False),  # 28x28
            nn.BatchNorm2d(g_channels),
            nn.ReLU(True),

            nn.Conv2d(g_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """
    DCGAN-ish discriminator for 28x28x1 -> scalar probability.
    """
    def __init__(self, d_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, d_channels, 4, 2, 1, bias=False),  # 14x14
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_channels, d_channels * 2, 4, 2, 1, bias=False),  # 7x7
            nn.BatchNorm2d(d_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_channels * 2, d_channels * 4, 3, 2, 1, bias=False),  # 4x4 (from 7 -> 4)
            nn.BatchNorm2d(d_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear((d_channels * 4) * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    if classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_grid_images(generator, fixed_noise, out_path, device):
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise.to(device)).cpu()
        # Convert from [-1,1] to [0,1]
        fake = (fake + 1.0) / 2.0
        grid = make_grid(fake, nrow=int(np.sqrt(fake.shape[0])), padding=2)
        save_image(grid, out_path)
    generator.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_every", type=int, default=1)
    args = parser.parse_args()

    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        save_every=args.save_every,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "weights"), exist_ok=True)

    device = torch.device(cfg.device)

    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # -> [-1, 1]
    ])

    ds = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    G = Generator(z_dim=cfg.latent_dim, g_channels=cfg.g_channels).to(device)
    D = Discriminator(d_channels=cfg.d_channels).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    optG = optim.Adam(G.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    optD = optim.Adam(D.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    fixed_noise = torch.randn(cfg.fixed_grid_n, cfg.latent_dim, 1, 1)

    # Save config for reproducibility
    with open(os.path.join(cfg.out_dir, "config.txt"), "w", encoding="utf-8") as f:
        for k, v in asdict(cfg).items():
            f.write(f"{k}: {v}\n")

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for real, _ in pbar:
            global_step += 1
            real = real.to(device)
            bsz = real.size(0)

            # -----------------------
            # Train Discriminator
            # -----------------------
            D.zero_grad(set_to_none=True)

            # Real labels = 1, fake labels = 0
            real_labels = torch.ones(bsz, 1, device=device)
            fake_labels = torch.zeros(bsz, 1, device=device)

            out_real = D(real)
            lossD_real = criterion(out_real, real_labels)

            noise = torch.randn(bsz, cfg.latent_dim, 1, 1, device=device)
            fake = G(noise)
            out_fake = D(fake.detach())
            lossD_fake = criterion(out_fake, fake_labels)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optD.step()

            # -----------------------
            # Train Generator
            # -----------------------
            G.zero_grad(set_to_none=True)

            out_fake2 = D(fake)
            # Generator wants discriminator to predict "real" (1)
            lossG = criterion(out_fake2, real_labels)
            lossG.backward()
            optG.step()

            pbar.set_postfix(lossD=float(lossD.item()), lossG=float(lossG.item()))

        # Save samples
        if epoch % cfg.save_every == 0:
            sample_path = os.path.join(cfg.out_dir, "samples", f"epoch_{epoch:03d}.png")
            save_grid_images(G, fixed_noise, sample_path, device)

        # Save checkpoints (optional but helpful)
        torch.save(G.state_dict(), os.path.join(cfg.out_dir, "weights", "G_latest.pt"))
        torch.save(D.state_dict(), os.path.join(cfg.out_dir, "weights", "D_latest.pt"))

    # Save final
    torch.save(G.state_dict(), os.path.join(cfg.out_dir, "weights", "G_final.pt"))
    torch.save(D.state_dict(), os.path.join(cfg.out_dir, "weights", "D_final.pt"))
    save_grid_images(G, fixed_noise, os.path.join(cfg.out_dir, "samples", "final.png"), device)

    print("Training complete.")
    print(f"Saved samples in: {os.path.join(cfg.out_dir, 'samples')}")
    print(f"Saved weights in: {os.path.join(cfg.out_dir, 'weights')}")


if __name__ == "__main__":
    main()
