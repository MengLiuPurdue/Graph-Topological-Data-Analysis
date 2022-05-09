import torch
from torchvision import transforms
import torch.nn.functional as F

def get_pil_transform(crop_size): 
    transf = transforms.Compose(
        [
            transforms.Resize(crop_size, interpolation=3),
            transforms.CenterCrop(224),
        ]
    )

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.4651, 0.4541, 0.4247],
                                    std=[0.2781, 0.2726, 0.2935])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf    