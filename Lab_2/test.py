import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as tf
import random
import matplotlib.pyplot as plt

# seed = 999 # for same results
seed = random.randint(1, 10000) # for variation
print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

img_size = 64
batch_size = 128
workers = 4
root="./data/CelebA"
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 5
# Learning rate for optimizers
lr = 2e-4
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

dataset = datasets.ImageFolder(root=root,
                           transform=tf.Compose([
                               tf.Resize(img_size),
                               tf.CenterCrop(img_size),
                               tf.ToTensor(),
                               tf.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
grid_img = torchvision.utils.make_grid(real_batch[0])
plt.imshow(grid_img.permute(1, 2, 0))
