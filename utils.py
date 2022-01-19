from typing import Dict
import torch
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import numpy as np


def visualize_results(img_in: torch.Tensor, img_out: torch.Tensor, layer_idx: int) -> None:
    img_out = img_out.data.numpy()[0].transpose(1, 2, 0)
    img_out = (img_out - img_out.min()) / (img_out.max() - img_out.min()) * 256
    img_out = img_out.astype(np.uint8)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_in.data.numpy()[0].transpose(1, 2, 0))
    ax[0].set_title("Input")

    ax[1].imshow(img_out)
    ax[1].set_title(f"Backprojection of strongest\n activation in layer {layer_idx}")
    plt.tight_layout()
    # plt.savefig("out.png", dpi=300)
    plt.show()


def get_image(file: str) -> torch.Tensor:
    image = Image.open(file)
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    X = transform(image).unsqueeze(dim=0)
    return X


def get_label_dict() -> Dict[int, str]:
    with open("imagenet1000_clsidx_to_labels.txt") as f:
        idx2label = eval(f.read())
    return idx2label
