"""
test file to see if running file works
"""
import random

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as tf
from torchvision import datasets

# seed = 999 # for same results
seed = random.randint(1, 10000)  # for variation
print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

IMG_SIZE = 64
BATCH_SIZE = 128
WORKERS = 4
ROOT = "./data/CelebA"
NZ = 100
# Size of feature maps in generator
NGF = 64
# Size of feature maps in discriminator
NDF = 64
# Number of training epochs
NUM_EPOCHS = 5
# Learning rate for optimizers
LR = 2e-4
# Beta1 hyperparam for Adam optimizers
BETA1 = 0.5

dataset = datasets.ImageFolder(
    root=ROOT,
    transform=tf.Compose(
        [
            tf.Resize(IMG_SIZE),
            tf.CenterCrop(IMG_SIZE),
            tf.ToTensor(),
            tf.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    ),
)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS
)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
grid_img = torchvision.utils.make_grid(real_batch[0])
plt.imshow(grid_img.permute(1, 2, 0))
