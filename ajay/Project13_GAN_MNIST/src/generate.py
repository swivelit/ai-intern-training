import os
import argparse
import torch
from torchvision.utils import save_image
from train import Generator  # reuse model definition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="outputs/weights/G_final.pt")
    parser.add_argument("--out_dir", type=str, default="outputs/generated")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    G = Generator(z_dim=args.latent_dim).to(device)
    G.load_state_dict(torch.load(args.weights, map_location=device))
    G.eval()

    with torch.no_grad():
        z = torch.randn(args.n, args.latent_dim, 1, 1, device=device)
        fake = G(z).cpu()
        fake = (fake + 1.0) / 2.0  # [-1,1] -> [0,1]

        # save as individual images
        for i in range(args.n):
            save_image(fake[i], os.path.join(args.out_dir, f"img_{i:03d}.png"))

    print(f"Saved {args.n} images to: {args.out_dir}")


if __name__ == "__main__":
    main()
