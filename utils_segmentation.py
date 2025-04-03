'''
@author: Loris Panza
'''
import cv2
from torchvision import transforms
import torch
import numpy as np

import matplotlib
from  matplotlib import pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights



def load_segmentation_model(device="cuda"):
    """
    Load a pretrained DeepLabV3 model for semantic segmentation.
    
    Args:
        device (str): The device to load the model on, either 'cuda' (GPU) or 'cpu'.
        
    Returns:
        model: The loaded DeepLabV3 segmentation model.
    """

    weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 # Vecchia versione supportata
    # oppure la versione più aggiornata:
    # weights = DeepLabV3_ResNet50_Weights.DEFAULT

    model = deeplabv3_resnet50(weights=weights).to(device)  # Uso corretto
    model.eval()
    return model


def filter_keypoints_by_mask(keypoints, mask, mask_number):
    """
    Filter keypoints that fall within the segmented region defined by the mask.
    
    Args:
        keypoints (np.ndarray): A list of keypoints (x, y) to be filtered.
        mask (np.ndarray): The segmentation mask where the regions of interest are marked.
        
    Returns:
        filtered_kpts (np.ndarray): The keypoints that are within the region of interest.
        deleted_kpts (np.ndarray): The keypoints that are outside the region of interest.
    """
    filtered_kpts = []
    deleted_kpts = []
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        # Check if the keypoint is within the desired region (here, class '4' is the target class).
        if mask[y, x] == mask_number:  # Assuming class '4' corresponds to the region of interest.
            filtered_kpts.append(kp)
        else: 
            deleted_kpts.append(kp)
    return np.array(filtered_kpts), np.array(deleted_kpts)


def filter_matched_keypoints_by_mask(keypoints0, keypoints1, mask0, mask1, mask_number):
    """
    Filter keypoints from two sets (keypoints0 and keypoints1) that fall within the segmented region 
    defined by two masks (mask0 and mask1).
    
    Args:
        keypoints0 (np.ndarray): A list of keypoints (x, y) for the first image.
        keypoints1 (np.ndarray): A list of keypoints (x, y) for the second image.
        mask0 (np.ndarray): The segmentation mask for the first image.
        mask1 (np.ndarray): The segmentation mask for the second image.
        
    Returns:
        filtered_kpts0 (np.ndarray): Keypoints from the first image that are within the region of interest.
        filtered_kpts1 (np.ndarray): Keypoints from the second image that are within the region of interest.
        deleted_kpts0 (np.ndarray): Keypoints from the first image that are outside the region of interest.
        deleted_kpts1 (np.ndarray): Keypoints from the second image that are outside the region of interest.
    """
    filtered_kpts0 = []
    filtered_kpts1 = []
    deleted_kpts0 = []
    deleted_kpts1 = []
    for kp0, kp1 in zip(keypoints0, keypoints1):
        x0, y0 = int(kp0[0]), int(kp0[1])
        x1, y1 = int(kp1[0]), int(kp1[1])
        # Check if both keypoints are within the desired region for both images.
        if mask0[y0, x0] == mask_number and mask1[y1, x1] == mask_number:
            filtered_kpts0.append(kp0)
            filtered_kpts1.append(kp1)
        else: 
            deleted_kpts0.append(kp0)
            deleted_kpts1.append(kp1)

    return np.array(filtered_kpts0), np.array(filtered_kpts1), np.array(deleted_kpts0), np.array(deleted_kpts1)


def overlay_mask(image, mask, path=None, alpha=0.5):
    """
    Overlay the segmentation mask on top of the original image for visualization.
    
    Args:
        image (np.ndarray or torch.Tensor): The original image to overlay the mask on.
        mask (np.ndarray): The segmentation mask to be overlaid on the image.
        alpha (float): The blending factor between the image and mask (default is 0.5).
        
    Returns:
        None: Displays the blended image with the mask.
    """
    # Ensure the image is a NumPy array. If it's a PyTorch tensor, convert it to NumPy.
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C).
        image = (image * 255).astype(np.uint8)  # Ensure the image is in uint8 format.

    # Convert the mask into a colorized version for visualization.
    mask_colored = cv2.applyColorMap((mask * 20).astype(np.uint8), cv2.COLORMAP_JET)

    # Ensure the mask has 3 channels (RGB) for blending.
    if len(mask_colored.shape) == 2:
        mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_GRAY2BGR)

    # Resize the mask to match the size of the image.
    mask_colored = cv2.resize(mask_colored, (image.shape[1], image.shape[0]))

    # Blend the original image and the mask with the specified alpha value.
    blended = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)

    # Display the result.
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Segmented Image Overlay")
    if path is not None:
        print("image saved at: ", path)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def segment_image(model, image, viz_path=None, device="cuda"):
    """
    Segment the input image using the given segmentation model and return a binary mask.
    
    Args:
        model (torch.nn.Module): The segmentation model to be used.
        image (np.ndarray): The image to be segmented.
        visualize (bool): If True, visualize the overlay of the mask on the image.
        device (str): The device to run the model on ('cuda' for GPU, 'cpu' for CPU).
        
    Returns:
        mask (np.ndarray): The resulting segmentation mask.
    """
    # Define a series of transformations to prepare the image for segmentation.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image.shape[1], image.shape[2])),  # Resize to model input size.
        transforms.ToTensor()
    ])
    
    # Transform the image into a tensor and send it to the specified device.
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Perform the forward pass to obtain the segmentation output.
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    
    # Convert the output to a binary mask where each pixel is assigned to the most likely class.
    mask = output.argmax(0).cpu().numpy()

    # Print the distribution of the classes in the mask.
    unique, counts = np.unique(mask, return_counts=True)
    #print("Mask Distribution:")
    #for u, c in zip(unique, counts):
    #    print(f"Class {u}: {c} pixels")
    
    # If visualize is True, overlay the mask on the image.
    
    overlay_mask(image, mask, viz_path)

    return mask