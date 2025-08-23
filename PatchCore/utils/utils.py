import random

import numpy as np
import torch
from PIL import ImageFilter
from sklearn import random_projection
from torchvision import transforms
from tqdm import tqdm


def set_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class KNNGaussianBlur(torch.nn.Module):
    def __init__(self, radius: int = 4):
        super().__init__()
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(self.unload(img[0] / map_max).filter(self.blur_kernel)) * map_max
        return final_map


def get_coreset_idx_randomp(z_lib, n=1000, eps=0.90, float16=True, force_cpu=False):
    """Returns n coreset idx for given z_lib.
    Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
    CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)
    Args:
        z_lib:      (n, d) tensor of patches.
        n:          Number of patches to select.
        eps:        Agression of the sparse random projection.
        float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
        force_cpu:  Force cpu, useful in case of GPU OOM.
    Returns:
        coreset indices
    """

    # print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
    try:
        transformer = random_projection.SparseRandomProjection(eps=0.9)
        z_lib = torch.tensor(transformer.fit_transform(z_lib))

        # print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
    except ValueError:
        print("   Error: could not project vectors. Please increase `eps`.")

    select_idx = 0
    last_item = z_lib[select_idx:select_idx + 1]
    coreset_idx = [torch.tensor(select_idx)]
    min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
    # The line below is not faster than linalg.norm, although i'm keeping it in for
    # future reference.
    # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

    if float16:
        last_item = last_item.half()
        z_lib = z_lib.half()
        min_distances = min_distances.half()
    if torch.cuda.is_available() and not force_cpu:
        last_item = last_item.to("cuda")
        z_lib = z_lib.to("cuda")
        min_distances = min_distances.to("cuda")

    for _ in tqdm(range(n - 1)):
        distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)  # broadcasting step
        min_distances = torch.minimum(distances, min_distances)  # iterative step
        select_idx = torch.argmax(min_distances)  # selection step

        # bookkeeping
        last_item = z_lib[select_idx:select_idx + 1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx.to("cpu"))
    return torch.stack(coreset_idx)
