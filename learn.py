from torchvision.datasets import CIFAR10 
from torchvision.datasets import ImageFolder

# dataset = CIFAR10(root='./data', train=True, download=True)
# image, label = dataset.__getitem__(8030)
# print(label)
# image.show()
dataset = ImageFolder(root = "animals/train")
print(dataset.class_to_idx)
image, label = dataset.__getitem__(3657)
print(label)
image.show()