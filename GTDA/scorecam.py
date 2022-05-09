#%%
"""
Created on Wed Apr 29 16:11:20 2020

@author: Haofan Wang - github.com/haofanwang
"""
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.cm as mpl_color_map
import copy
from tqdm import tqdm

class CamExtractorBase(object):
    """
        Extracts cam features from the model
    """
    def __init__(self, model):
        self.model = model
    
    def forward_pass_on_convolutions(self):
        raise NotImplementedError()
    
    def forward_pass(self):
        raise NotImplementedError()

class CamExtractor(CamExtractorBase):
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer=11):
        super().__init__(model)
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x

class CamExtractorResNet50(CamExtractorBase):
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer="layer4"):
        super().__init__(model)
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        with torch.set_grad_enabled(False):
            for module_pos, module in self.model._modules.items():
                if module_pos == "fc":
                    x = torch.flatten(x, 1)
                x = module(x)  # Forward
                if module_pos == self.target_layer:
                    conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = F.softmax(x)
        return conv_output, x

class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, extractor):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = extractor

    def generate_cam(self, input_image, target_class=None, crop_size=224, batch_size=128):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.cpu().numpy(),1)
        all_cam = []
        all_new_img = []
        for k in tqdm(range(conv_output.shape[0])):
            # Get convolution outputs
            target = conv_output[k]
            # Create empty numpy array for cam
            cam = np.ones(target.shape[1:], dtype=np.float32)
            # Multiply each weight with its conv output and then, sum
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(target,0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
            maxs = torch.amax(saliency_map,dim=(0,2,3))
            mins = torch.amin(saliency_map,dim=(0,2,3))
            selected_channels = torch.nonzero(maxs-mins).flatten()
            saliency_map = saliency_map[:,selected_channels,:,:]
            maxs = maxs[selected_channels]
            mins = mins[selected_channels]
            # Scale between 0-1
            norm_saliency_map = (saliency_map - mins[None,:,None,None]) / (maxs-mins)[None,:,None,None]
            all_new_img = norm_saliency_map[0][:,None]*input_image[k]
            # Get the target score
            for start_i in range(0,all_new_img.shape[0],batch_size):
                end_i = min(start_i+batch_size,all_new_img.shape[0])
                w = F.softmax(self.extractor.forward_pass(all_new_img[start_i:end_i])[1],dim=1)[:,target_class[k]]
                cam += torch.sum(w.data[:,None,None] * target.data[start_i:end_i],0).cpu().numpy()
            cam = np.maximum(cam,0)
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            cam = np.uint8(Image.fromarray(cam).resize((input_image[[k]].shape[2],
                        input_image[[k]].shape[3]), Image.ANTIALIAS))/255
            all_cam.append(cam)
        return all_cam

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image

#%%
# Get params
# target_example = 0  # Snake
# (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
#     get_example_params(target_example)
# # Score cam
# pretrained_model = pretrained_model.to('cuda')
# extractor = CamExtractor(pretrained_model,target_layer=11)
# score_cam = ScoreCam(pretrained_model,extractor)
# # Generate cam mask
# prep_img = prep_img.to('cuda')
# cam = score_cam.generate_cam(prep_img, target_class)
# # Save mask
# save_class_activation_images(original_image, cam, file_name_to_export)
# print('Score cam completed')


# %%
