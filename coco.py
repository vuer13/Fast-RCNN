from PIL import Image
import torch
from torchvision.transforms import functional as F

class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image) # Transforms image to tensor
        return image, target