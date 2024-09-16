import torch
from rembg import remove
from PIL import Image
import torch
import numpy as np
import random
from icecream import ic
from datetime import datetime
import os

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Common upscale function to resize image2 to match dimensions of image1
def common_upscale(image, width, height, mode="bilinear", align_corners=False):
    return torch.nn.functional.interpolate(image, size=(height, width), mode=mode, align_corners=align_corners)

def pad_images(images):
    max_height = max(image.size[1] for image in images)
    max_width = max(image.size[0] for image in images)
    padded_images = []

    for image in images:
        width, height = image.size
        
        # Scale the image to minimize padding
        scale_factor = min(max_width / width, max_height / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        scaled_image = image.resize((new_width, new_height), Image.BILINEAR)
        
        # Create a new image with the maximum size and paste the scaled image into the center
        new_image = Image.new("RGB", (max_width, max_height), (0, 0, 0))
        paste_x = (max_width - new_width) // 2
        paste_y = (max_height - new_height) // 2
        new_image.paste(scaled_image, (paste_x, paste_y))
        
        padded_images.append(new_image)
    
    return padded_images

class JHK_Utils_LoadImagesFromPath:
    NODE_NAME = "Load Image From Path"
    CATEGORY = "JHK_utils/load"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "result"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": '', "multiline": False}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def result(self, image_path, **other):
        # Allowed image extensions
        allowed_extensions = ('.png', '.jpg', '.jpeg')

        # Get list of all image files
        image_files = [f for f in os.listdir(image_path) if f.lower().endswith(allowed_extensions)]

        # Raise an error if no images are found
        if not image_files:
            raise ValueError("No images found in the specified path.")

        # Shuffle and select up to 10 images
        random.shuffle(image_files)
        selected_files = image_files[:6]

        # List to store loaded images
        images = []

        # Iterate over the selected files
        for filename in selected_files:
            full_path = os.path.join(image_path, filename)
            image = Image.open(full_path).convert('RGB')  # Open and convert to RGB
            images.append(image)
        
        # Pad images to match the dimensions of the largest image
        padded_images = pad_images(images)
        
        # Convert images to tensors
        tensor_images = [pil2tensor(image) for image in padded_images]

        # Stack images into a batch
        batch_images = torch.cat(tensor_images, dim=0)

        return (batch_images,)
        # return None
    
class JHK_Utils_LoadEmbed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = ("EMBEDS", )
    FUNCTION = "load"
    CATEGORY = "JHK_utils/embed"

    def load(self, path):
        return (torch.load(path).cpu(), )

class JHK_Utils_string_merge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": '', "multiline": False}),
            },
            "optional": {
                "string2": ("STRING", {"default": '', "multiline": False}),
                "string3": ("STRING", {"default": '', "multiline": False}),
                "string4": ("STRING", {"default": '', "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", )
    FUNCTION = "merge"
    CATEGORY = "JHK_utils/string"
    
    def merge(self, string1, string2='', string3='', string4=''):
        # Split each string into a set of tags
        set1 = set(string1.split(", "))
        set2 = set(string2.split(", ")) if string2 else set()
        set3 = set(string3.split(", ")) if string3 else set()
        set4 = set(string4.split(", ")) if string4 else set()
        
        # Combine all sets into one to remove duplicates
        combined_set = set1.union(set2).union(set3).union(set4)
        
        # Convert the set back to a sorted list and then to a single string
        merged_string = ', '.join(sorted(combined_set))
        
        # Print combined set and merged string for debugging
        # ic(combined_set)  # Use ic for introspection if ic is imported, else use print
        # ic(merged_string)
        
        # Return the result in the specified format
        return {"ui": {"text": merged_string}, "result": (merged_string,)}

class JHK_Utils_string_filter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": '', "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", )
    FUNCTION = "merge"
    CATEGORY = "JHK_utils/string"
    
    def merge(self, string1):
        # Split each string into a set of tags
        set1 = set(string1.split(", "))
        
        extract_list = ["skin", "bald", "hair", "male", "female", "eye", "nose", "mouth", "skin", "beard"]
        # Filter in only desired tags
        filtered_set = {tag for tag in set1 if any(substring in tag for substring in extract_list)}
        
        # # Add ", completely bald" if "bald" is in the tags
        # if any("bald" in tag for tag in filtered_set):
        #     filtered_set.add("Bald headed")
        #     filtered_set.add("Hairless")
        #     filtered_set.add("Shaven headed")
        #     filtered_set.add("Smooth scalped")
        #     filtered_set.add("no hair")

        # Convert the set back to a sorted list and then to a single string
        merged_string = ', '.join(sorted(filtered_set))
        
        # Return the result in the specified format
        return {"ui": {"text": merged_string}, "result": (merged_string,)}
    
class JHK_Utils_ImageRemoveBackground:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_background"
    CATEGORY = "JHK_utils/image"

    def remove_background(self, image):
        image = pil2tensor(remove(tensor2pil(image)))
        return (image,)

class JHK_Utils_RandomImageSelector:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"default": 0, "min": 0,  "max": 0xffffffffffffffff, "step": 1} ),  # Input is a batch of images as a single tensor
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select_random_image"
    CATEGORY = "JHK_utils/image"

    def select_random_image(self, images):
        """Selects a random image from the provided batch of images."""
        batch_size = images.size(0)
        random.seed(datetime.now().timestamp())
        # rand_index = random.randint(0, batch_size - 1)
        rand_index = int( datetime.now().timestamp() ) % batch_size
        image = images[rand_index]
        image = image.unsqueeze(0)
        return (image, )

class JHK_Utils_LargestImageSelector:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1} ),  # Input is a batch of images as a single tensor
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select_largest_image"
    CATEGORY = "JHK_utils/image"

    def select_largest_image(self, images):
        """Selects the largest image from the provided batch of images."""
        # Initialize variables to store the index and size of the largest image
        largest_image_index = 0
        largest_image_size = (0, 0)
        
        # Iterate through the batch of images
        for i in range(images.size(0)):
            _, height, width = images[i].shape
            if (height * width) > (largest_image_size[0] * largest_image_size[1]):
                largest_image_size = (height, width)
                largest_image_index = i
        
        # Select the largest image
        largest_image = images[largest_image_index]
        largest_image = largest_image.unsqueeze(0)  # Add batch dimension
        
        return (largest_image, )

class JHK_Utils_SelectSingleImageFromPath:
    NODE_NAME = "Select Single Image From Path"
    CATEGORY = "JHK_utils/load"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "result"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": '', "multiline": False}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def result(self, image_path, **other):
        # Allowed image extensions
        allowed_extensions = ('.png', '.jpg', '.jpeg')

        # Get list of all image files
        image_files = [f for f in os.listdir(image_path) if f.lower().endswith(allowed_extensions)]

        # Raise an error if no images are found
        if not image_files:
            raise ValueError("No images found in the specified path.")

        # Shuffle and select one image
        random.shuffle(image_files)
        selected_file = image_files[0]

        # Load the selected image
        full_path = os.path.join(image_path, selected_file)
        image = Image.open(full_path).convert('RGB')  # Open and convert to RGB
        
        # Convert image to tensor
        image_tensor = pil2tensor(image)

        return (image_tensor,)
    
NODE_CLASS_MAPPINGS = {
    # Main Apply Nodes
    "JHK_Utils_LoadEmbed": JHK_Utils_LoadEmbed,
    "JHK_Utils_string_merge":JHK_Utils_string_merge,
    "JHK_Utils_ImageRemoveBackground": JHK_Utils_ImageRemoveBackground,
    "JHK_Utils_RandomImageSelector": JHK_Utils_RandomImageSelector,
    "JHK_Utils_LoadImageFromPath":JHK_Utils_LoadImagesFromPath,
    "JHK_Utils_string_filter":JHK_Utils_string_filter,
    "JHK_Utils_LargestImageSelector":JHK_Utils_LargestImageSelector,
    "JHK_Utils_SelectSingleImageFromPath":JHK_Utils_SelectSingleImageFromPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Main Apply Nodes
    "JHK_Utils_LoadEmbed": "JHK_Utils_LoadEmbed",
    "JHK_Utils_string_merge": "JHK_Utils_string_merge",
    "JHK_Utils_ImageRemoveBackground": "JHK_Utils_ImageRemoveBackground",
    "JHK_Utils_RandomImageSelector":"JHK_Utils_RandomImageSelector",
    "JHK_Utils_LoadImageFromPath":"JHK_Utils_LoadImageFromPath",
    "JHK_Utils_string_filter":"JHK_Utils_string_filter",
    "JHK_Utils_LargestImageSelector":"JHK_Utils_LargestImageSelector",
    "JHK_Utils_SelectSingleImageFromPath":"JHK_Utils_SelectSingleImageFromPath",
}


# pip install rembg[gpu]
# https://github.com/cubiq/ComfyUI_FaceAnalysis.git

# Example usage
if __name__ == "__main__":
    loader = JHK_Utils_LoadImagesFromPath()
    batch_images = loader.result('./your/test_path/')
    print(batch_images[0].shape)  # This should print the shape of the first image tensor in the batch


# def pil2tensor(image):
#     # Convert PIL image to numpy array, normalize, convert to tensor
#     tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
#     # Add a batch dimension if needed (based on your processing pipeline)
#     return tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
