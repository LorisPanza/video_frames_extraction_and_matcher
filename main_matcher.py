"""
This script performs image matching using a specified matcher model. It processes pairs of input images,
detects keypoints, matches them, and performs RANSAC to find inliers. The results, including visualizations
and metadata, are saved to the specified output directory.
"""

import torch
import argparse

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
import re
import cv2


from matching.utils import get_image_pairs_paths, get_model_folders, load_torch_save, pair_images_in_folder
from matching import get_matcher, available_models
from matching.viz import plot_matches, plot_barplot_curr_folder
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms




def load_segmentation_model(device="cuda"):
    """Load a pretrained DeepLabV3 model for segmentation."""
    model = deeplabv3_resnet50(pretrained=True).to(device)
    model.eval()
    return model


def filter_keypoints_by_mask(keypoints, mask):
    """Filter keypoints that fall within the segmented region."""
    filtered_kpts = []
    deleted_kpts = []
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        if mask[y, x] == 4 :  # Assuming the ROI has nonzero values, 4 is the class boat
            filtered_kpts.append(kp)
        else: 
            deleted_kpts.append(kp)
    return np.array(filtered_kpts), np.array(deleted_kpts)


def filter_matched_keypoints_by_mask(keypoints0, keypoints1, mask0, mask1):
    """Filter keypoints that fall within the segmented region."""
    filtered_kpts0 = []
    filtered_kpts1 = []
    deleted_kpts0 = []
    deleted_kpts1 = []
    for kp0, kp1 in zip(keypoints0, keypoints1):
        x0, y0 = int(kp0[0]), int(kp0[1])
        x1, y1 = int(kp1[0]), int(kp1[1])
        if mask0[y0, x0] == 4 and mask1[y1, x1] == 4:  # Assuming the ROI has nonzero values, 4 is the class boat
            filtered_kpts0.append(kp0)
            filtered_kpts1.append(kp1)
        else: 
            deleted_kpts0.append(kp0)
            deleted_kpts1.append(kp1)

    return np.array(filtered_kpts0), np.array(filtered_kpts1), np.array(deleted_kpts0), np.array(deleted_kpts1)


def overlay_mask(image, mask, alpha=0.5):
    """Overlay segmentation mask on the original image."""
    # Ensure image is a NumPy array
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
        image = (image * 255).astype(np.uint8)  # Ensure uint8

    # Convert mask to color (random colors for different classes)
    mask_colored = cv2.applyColorMap((mask * 20).astype(np.uint8), cv2.COLORMAP_JET)

    # Ensure mask has 3 channels
    if len(mask_colored.shape) == 2:
        mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_GRAY2BGR)

    # Resize mask to match image size
    mask_colored = cv2.resize(mask_colored, (image.shape[1], image.shape[0]))

    # Blend original image and mask
    blended = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)

    # Display the result
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Segmented Image Overlay")
    plt.show()


def segment_image(model, image,visualize=False, device="cuda"):
    """Segment the image using the given model and return a binary mask."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image.shape[1], image.shape[2])),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    
    mask = output.argmax(0).cpu().numpy()

    # Print the unique values and their counts
    unique, counts = np.unique(mask, return_counts=True)
    print("Mask Distribution:")
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c} pixels")
    
    if visualize:
        overlay_mask(image, mask)

    return mask


def masking_result(result, mask0 = None, mask1 = None):

    if(mask0 is not None and mask1 is not None):

        matched_kpts0, matched_kpts1 = result["matched_kpts0"], result["matched_kpts1"]
        all_kpts0, all_kpts1 = result["all_kpts0"], result["all_kpts1"]
        inlier_kpts0, inlier_kpts1 = result["inlier_kpts0"], result["inlier_kpts1"]

        # Filter keypoints based on segmentation masks - ALL
        all_filtered_kpts0, deleted_kpts0 = filter_keypoints_by_mask(all_kpts0, mask0)
        all_filtered_kpts1, deleted_kpts1 = filter_keypoints_by_mask(all_kpts1, mask1)

        # Filter keypoints based on segmentation masks - Matched
        matched_filtered_kpts0, matched_filtered_kpts1, deleted_kpts0, deleted_kpts1  = filter_matched_keypoints_by_mask(matched_kpts0, matched_kpts1, mask0, mask1)

        # Filter keypoints based on segmentation masks - Matched
        inlier_filtered_kpts0, inlier_filtered_kpts1, deleted_kpts0, deleted_kpts = filter_matched_keypoints_by_mask(inlier_kpts0, inlier_kpts1, mask0, mask1)

        # Create a new result dictionary with the filtered keypoints
        filtered_result = result.copy()  # Start with a copy of the original result

        # Update the keypoints with filtered ones
        filtered_result["all_kpts0"] = all_filtered_kpts0
        filtered_result["all_kpts1"] = all_filtered_kpts1

        # Update the matched keypoints with filtered ones
        filtered_result["matched_kpts0"] = matched_filtered_kpts0
        filtered_result["matched_kpts1"] = matched_filtered_kpts1

        # Update the inlier keypoints with filtered ones
        filtered_result["inlier_kpts0"] = inlier_filtered_kpts0
        filtered_result["inlier_kpts1"] = inlier_filtered_kpts1

        # Find indices of kept keypoints
        if len(all_filtered_kpts0!=0):
            kept_indices = [i for i, kp in enumerate(all_kpts0) if kp in all_filtered_kpts0]
            filtered_result["all_desc0"] = result["all_desc0"][kept_indices]
        else:
            filtered_result["all_desc0"] = np.array([])
            

        if len(all_filtered_kpts1!=0):
            kept_indices = [i for i, kp in enumerate(all_kpts1) if kp in all_filtered_kpts1]
            filtered_result["all_desc1"] = result["all_desc1"][kept_indices]
        else:
            filtered_result["all_desc1"] = np.array([])

        filtered_result["num_inliers"] = len(inlier_filtered_kpts0)
        filtered_result["H"] = result["H"]

        return filtered_result
    else:
        return result

def main(args):
    image_size = [args.im_size, args.im_size]
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Choose a matcher
    matcher = get_matcher(args.matcher, device=args.device, max_num_keypoints=args.n_kpts)
    pairs_of_paths, folders_name = get_image_pairs_paths(args.input) #changed the return arguments

    for i, (img0_path, img1_path) in enumerate(pairs_of_paths):

        image0 = matcher.load_image(img0_path, resize=image_size)
        image1 = matcher.load_image(img1_path, resize=image_size)
        result = matcher(image0, image1)
        
        out_str = f"Paths: {str(img0_path), str(img1_path)}. Found {result['num_inliers']} inliers after RANSAC. "

        if not args.no_viz:
            viz_path = args.out_dir / f"output_{i}_matches.jpg"
            plot_matches(image0, image1, result, save_path=viz_path)
            out_str += f"Viz saved in {viz_path}. "

        result["img0_path"] = img0_path
        result["img1_path"] = img1_path
        result["matcher"] = args.matcher
        result["n_kpts"] = args.n_kpts
        result["im_size"] = args.im_size

        dict_path = args.out_dir / f"output_{i}_result.torch"
        torch.save(result, dict_path)
        out_str += f"Output saved in {dict_path}"
        print(out_str)

'''
@author: Loris
result is a dict with those arguments: num_inliers, all_kpts0/1, matched_kpts0/1
'''
def robustness_analysis(out_dir, plot=True):
    """
    Analyzes the robustness of model outputs by computing inlier ratios from saved results.
    
    Args:
        out_dir (str): Path to the directory containing model output folders.
        plot (bool, optional): If True, generates bar plots of the inlier ratios. Default is False.
    
    Returns:
        None
    """
    
    # Retrieve all model output folders inside the given directory
    model_output_folders = get_model_folders(out_dir)  
    dict_result_analysis = {}  # Dictionary to store mean inlier ratios per folder
    pair_pattern = re.compile(r"(pair_\d+)")  # Regex pattern to extract image pair names
    dict_performances = {}  # Stores performance values for each folder
    performances_order = []  # Keeps track of the order in which performances are stored

    # Iterate through each model output folder
    for out_folder in model_output_folders:
        model_folder_name = os.path.split(out_folder)[-1]  # Extract folder name
        performances_order.append(out_folder)  # Keep track of processing order
        
        # Iterate through all result files in the current folder
        folder_image_names = Path(out_folder).glob("*")
        for folder_image in folder_image_names:
            out_folder_name = os.path.split(folder_image)[-1]  # Get the sub-folder name
            results_curr_folder = load_torch_save(folder_image)  # Load the stored results
            
            print(f"Extracted {len(results_curr_folder)} results from {model_folder_name}-{out_folder_name}.")
            
            ratio_arr = []  # Stores inlier ratios for the current folder
            curr_images_name = []  # Stores image pair names
            
            # Initialize the performance dictionary entry for the current folder if not already present
            if out_folder_name not in dict_performances.keys():
                print(f"Initializing {out_folder_name}")
                dict_performances[out_folder_name] = []

            # Process each stored result file
            for str_res in results_curr_folder:
                res = torch.load(str_res, weights_only=False)  # Load the saved torch file
                
                # Ensure that the number of matched keypoints is consistent between images
                assert len(res["matched_kpts1"]) == len(res["matched_kpts0"])

                # Compute the ratio of inliers over total matches (should be ideally 1)
                curr_ratio = res["num_inliers"] / len(res["matched_kpts0"])
                ratio_arr.append(curr_ratio)

                # Extract image pair name using regex
                match = pair_pattern.search(str_res)
                imgs_name = match.group(1)
                curr_images_name.append(imgs_name)

            # Compute the mean inlier ratio for the current folder
            mean_ratio_arr = round(np.mean(ratio_arr), 2)
            dict_performances[out_folder_name].append(mean_ratio_arr)
            
            print(f"Mean ratio {curr_images_name}: {ratio_arr}")

            # Plot the results if required
            if plot:
                plot_barplot_curr_folder(ratio_arr, curr_images_name, out_folder_name)
            
            print(f"{out_folder_name} ratio: {mean_ratio_arr:0.2f}.")
            dict_result_analysis[out_folder_name] = mean_ratio_arr  # Store mean ratio

    # Print the order of processed folders
    print(f"Order of performances dict: {performances_order}")
    
    # Print computed mean ratios for each folder
    for key, val in dict_performances.items():
        print(f"Mean array {key}: {val}")


def extract_keypoints(args, segmentation_model=None):
    image_size = [args.im_size, args.im_size]
    subfolder_mask = {}
    pair_dict = {}
    gt_flag = False

    #Check if exists the out dir
    args.out_dir.mkdir(exist_ok=True, parents=True)
    
    # Choose a matcher
    matcher = get_matcher(args.matcher, device=args.device, max_num_keypoints=args.n_kpts)
    
    model_folders = get_model_folders(args.input) #ex: [gt, hat, lq]

    for model in model_folders:

        # output/model creation
        model_name = os.path.split(model)[1]
        out_model_path = os.path.join(args.out_dir,model_name)
        os.makedirs(name=out_model_path, exist_ok=True)

        # input_dir/model/image_folders
        subfolders = os.listdir(model) #ex: [test_pipeline, etc..]

        for subfolder in subfolders: #ex: [subfolder/image_pair1.png, image_pair2.png] , in each subfolders onlyt two images
            # intializing mask to None in case there is no gt folder
            mask0 = None
            mask1 = None

            subfolder_path = os.path.join(model, subfolder)
            pairs_of_paths, folder_names = get_image_pairs_paths(subfolder_path) #returns only two elements in the iterator

            # input_dir/model/image_folders/pair_folder/...png
            for i, (img0_path, img1_path) in enumerate(pairs_of_paths): #[pair0, pair1,...]
                
                pair_folder_name = os.path.split(folder_names[i])[1]

                image_0_name = Path(img0_path).stem
                img_1_name = Path(img1_path).stem
                image0 = matcher.load_image(img0_path, resize=image_size)
                image1 = matcher.load_image(img1_path, resize=image_size)


                if model_name == "gt":
                    gt_flag = True
                    # Segment both images
                    mask0 = segment_image(segmentation_model, image0, False ,args.device)
                    mask1 = segment_image(segmentation_model, image1, False, args.device)
                    pair_dict[pair_folder_name] = (mask0, mask1)
                    subfolder_mask[subfolder] = pair_dict


                elif(gt_flag):
                        # retrieve the mask from the gt folder
                        mask0 = subfolder_mask[subfolder][pair_folder_name][0]
                        mask1 = subfolder_mask[subfolder][pair_folder_name][1]

                result = matcher(image0, image1)

                filtered_result = masking_result(result, mask0, mask1)

                out_str = f"Paths: {str(img0_path), str(img1_path)}. Found {filtered_result['num_inliers']} inliers after RANSAC. "
                #curr_folder_name = os.path.split(folder_names[i])[-1]

                #output_dir/model/image_folders/   creation
                curr_path_folder_image = os.path.join(out_model_path, subfolder)
                os.makedirs(name=curr_path_folder_image, exist_ok=True)

                if not args.no_viz and filtered_result["num_inliers"] != 0: 
                    #output_dir/model/image_folders/result.png-torch
                    viz_path = os.path.join(curr_path_folder_image, f"output_{subfolder}_{image_0_name}_{img_1_name}_matches.jpg") #str(out_model_path) / f"output_{i}_matches.jpg"
                    plot_matches(image0, image1, filtered_result, save_path=viz_path)
                    out_str += f"Viz saved in {viz_path}. "

                filtered_result["img0_path"] = img0_path
                filtered_result["img1_path"] = img1_path
                filtered_result["matcher"] = args.matcher
                filtered_result["n_kpts"] = args.n_kpts
                filtered_result["im_size"] = args.im_size

                dict_path = os.path.join(curr_path_folder_image, f"output_{subfolder}_{image_0_name}_{img_1_name}_result.torch")
                torch.save(filtered_result, dict_path)
                out_str += f"Output saved in {out_model_path}"
                print(out_str)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Image Matching Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Choose matcher
    parser.add_argument(
        "--matcher",
        type=str,
        default="sift-lg",
        choices=available_models,
        help="choose your matcher",
    )

    # Hyperparameters shared by all methods:
    parser.add_argument("--im_size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--n_kpts", type=int, default=2048, help="max num keypoints")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--no_viz", action="store_true", help="avoid saving visualizations")

    parser.add_argument(
        "--input",
        type=str,
        default=None, #"assets/example_pairs"
        help="path to either (1) dir with dirs with image pairs or (2) txt file with two image paths per line",
    )
    parser.add_argument("--out_dir", type=Path, default=None, help="path where outputs are saved") # frames_matched\name_folder_matched 
    #parser.add_argument("--sr_robustness", action="store_true", help="Apply the matcher and make the comparison and create a metric among swin, hat and pipeline output to prove the robustness of the model.") # it takes as input a path like: input_dir/models/image_folder/pair_folder/image.png
    parser.add_argument("--analysis",action="store_true", help="making the analysis of robustness without the inference process") 
    parser.add_argument("--extract_keypoints", action="store_true", help="making the analysis of robustness without the inference process") 
    parser.add_argument("--images_to_be_paired", type=Path, default=None, help ="pair the image in folders by the name.") # frames_source\salient_frames_name_folder that could be salient_frames_name_folder/models/list_of_images.png or salient_frames_name_folder/models/image_folders/list_of_images.png

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"outputs_{args.matcher}")

    return args


if __name__ == "__main__":
    args = parse_args()
    print(torch.cuda.is_available())
    print(args)

    if(not(args.images_to_be_paired is None)):
        print("Creating folder")
        # args.out to save matched frames
        assert not(args.out_dir is None)
        pair_images_in_folder(args.images_to_be_paired, args.out_dir)

    if(args.analysis):
        assert not(args.out_dir is None)
        robustness_analysis(args.out_dir)

    if(args.extract_keypoints):
        #args in to consider the input folder, args out to write the ouput
        assert not(args.input is None and args.out is None)
        seg_model = load_segmentation_model(device=args.device)
        extract_keypoints(args, seg_model)
    
    

