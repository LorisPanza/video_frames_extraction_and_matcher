import torch
import argparse
from pathlib import Path
import os
import numpy as np
import re
from matching.utils import get_image_pairs_paths, get_model_folders, load_torch_save
from matching import get_matcher, available_models
from matching.viz import plot_matches, plot_barplot_curr_folder
from utils_segmentation import load_segmentation_model, segment_image, filter_keypoints_by_mask, filter_matched_keypoints_by_mask, overlay_mask
import json




def masking_result(result, mask_number, mask0=None, mask1=None):
    """
    Filters keypoints based on segmentation masks.

    Args:
        result (dict): Dictionary containing keypoints, matches, and descriptors.
        mask_number (int): The class number in the segmentation mask used to filter keypoints.
        mask0 (ndarray, optional): Segmentation mask for the first image. Defaults to None.
        mask1 (ndarray, optional): Segmentation mask for the second image. Defaults to None.

    Returns:
        dict: Updated result dictionary with keypoints filtered based on the segmentation masks.
    """

    # Ensure masks are provided before proceeding with filtering
    if mask0 is not None and mask1 is not None and mask_number is not None:

        # Extract keypoints from the result dictionary
        matched_kpts0, matched_kpts1 = result["matched_kpts0"], result["matched_kpts1"]
        all_kpts0, all_kpts1 = result["all_kpts0"], result["all_kpts1"]
        inlier_kpts0, inlier_kpts1 = result["inlier_kpts0"], result["inlier_kpts1"]

        # Filter all detected keypoints based on the segmentation masks
        all_filtered_kpts0, deleted_kpts0 = filter_keypoints_by_mask(all_kpts0, mask0, mask_number=mask_number)
        all_filtered_kpts1, deleted_kpts1 = filter_keypoints_by_mask(all_kpts1, mask1, mask_number=mask_number)

        # Filter matched keypoints based on segmentation masks
        matched_filtered_kpts0, matched_filtered_kpts1, deleted_kpts0, deleted_kpts1 = filter_matched_keypoints_by_mask(
            matched_kpts0, matched_kpts1, mask0, mask1, mask_number=mask_number
        )

        # Filter inlier keypoints (keypoints that survived RANSAC filtering)
        inlier_filtered_kpts0, inlier_filtered_kpts1, deleted_kpts0, deleted_kpts = filter_matched_keypoints_by_mask(
            inlier_kpts0, inlier_kpts1, mask0, mask1, mask_number=mask_number
        )

        # Create a copy of the original result dictionary to store the filtered keypoints
        filtered_result = result.copy()

        # Update the result dictionary with the filtered keypoints
        filtered_result["all_kpts0"] = all_filtered_kpts0
        filtered_result["all_kpts1"] = all_filtered_kpts1

        filtered_result["matched_kpts0"] = matched_filtered_kpts0
        filtered_result["matched_kpts1"] = matched_filtered_kpts1

        filtered_result["inlier_kpts0"] = inlier_filtered_kpts0
        filtered_result["inlier_kpts1"] = inlier_filtered_kpts1

        # Ensure the descriptors are updated based on the filtered keypoints
        if len(all_filtered_kpts0) != 0:
            # Keep descriptors only for the keypoints that were not removed
            kept_indices = [i for i, kp in enumerate(all_kpts0) if kp in all_filtered_kpts0]
            filtered_result["all_desc0"] = result["all_desc0"][kept_indices]
        else:
            # If no keypoints remain, return an empty descriptor array
            filtered_result["all_desc0"] = np.array([])

        if len(all_filtered_kpts1) != 0:
            kept_indices = [i for i, kp in enumerate(all_kpts1) if kp in all_filtered_kpts1]
            filtered_result["all_desc1"] = result["all_desc1"][kept_indices]
        else:
            filtered_result["all_desc1"] = np.array([])

        # Update the number of inliers after filtering
        filtered_result["num_inliers"] = len(inlier_filtered_kpts0)

        # Preserve the homography matrix (unchanged)
        filtered_result["H"] = result["H"]

        return filtered_result

    else:
        # If masks are not provided, return the original result without filtering
        return result



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


def extract_keypoints(args, mask_number, segmentation_model, gt_folder=None):
    """
    Extract keypoints from image pairs, apply segmentation masks, and save the results.

    Args:
        args: The arguments containing various configurations (like image size, matcher type, etc.).
        segmentation_model: A pre-trained segmentation model (default is None, used for the 'gt' folder).
        mask_number: The class number used to filter keypoints.

    Returns:
        None: Saves the results in the specified output directory.
    """

    image_size = [args.im_size, args.im_size]  # Set the image size for resizing images
    subfolder_mask = {}  # Dictionary to store masks for each subfolder
    gt_flag = False  # Flag to track if we are processing the 'gt' folder

    # Ensure the output directory exists
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Initialize the matcher (used for finding keypoints in the images)
    matcher = get_matcher(args.matcher, device=args.device, max_num_keypoints=args.n_kpts)

    # Get the folders in the input directory (e.g., 'gt', 'hat', 'lq')
    model_folders = get_model_folders(args.input)
    model_names = [os.path.split(single_model_path)[1] for single_model_path in model_folders]

    assert gt_folder in model_names or gt_folder == None
    
    if(gt_folder is not None):
        # If a gt_folder is specified, remove it from the list of model folders
        gt_folder_path = os.path.join(args.input, gt_folder)
        print(f"GT folder path: {gt_folder_path}")
        print("Model folders before removing gt folder: ", model_folders)
        model_folders.remove(Path(gt_folder_path))
        print(f"Removing {gt_folder_path} from the list of folders to be processed.")
        model_folders.insert(0, gt_folder_path)  # Add the gt_folder to the beginning of the list
        print("Model folders after removing gt folder and put it as first: ", model_folders)

    for model in model_folders:
        # Extract model name and set up the output folder
        model_name = os.path.split(model)[1]
        out_model_path = os.path.join(args.out_dir, model_name)
        os.makedirs(name=out_model_path, exist_ok=True)

        # Get the subfolders inside the model directory (e.g., 'test_pipeline', 'etc.')
        subfolders = os.listdir(model)

        for subfolder in subfolders:
            mask0 = None  # Initialize mask0
            mask1 = None  # Initialize mask1

            # Path to the subfolder containing image pairs
            subfolder_path = os.path.join(model, subfolder)
            pairs_of_paths, folder_names = get_image_pairs_paths(subfolder_path)  # Get pairs of image paths

            for i, (img0_path, img1_path) in enumerate(pairs_of_paths):  # Iterate through image pairs
                pair_folder_name = os.path.split(folder_names[i])[1]  # Get the name of the image pair folder

                # Get the image names (without extensions)
                image_0_name = Path(img0_path).stem
                img_1_name = Path(img1_path).stem

                # Create the output folder for the current subfolder and image pair
                curr_path_folder_image = os.path.join(out_model_path, subfolder)
                os.makedirs(name=curr_path_folder_image, exist_ok=True)
                viz_path_matching = os.path.join(curr_path_folder_image, f"output_{subfolder}_{image_0_name}_{img_1_name}_matches.jpg")
                viz_path_masking_0 = os.path.join(curr_path_folder_image, f"output_{subfolder}_{image_0_name}_{img_1_name}_mask_0.jpg")
                viz_path_masking_1 = os.path.join(curr_path_folder_image, f"output_{subfolder}_{image_0_name}_{img_1_name}_mask_1.jpg")

                
                # Load the images (resize them to the specified size)
                image0 = matcher.load_image(img0_path, resize=image_size)
                image1 = matcher.load_image(img1_path, resize=image_size)

                # If processing the 'gt' folder, segment both images and save the masks
                if model_name == gt_folder:
                    gt_flag = True
                    mask0 = segment_image(segmentation_model, image0, viz_path_masking_0, args.device)
                    mask1 = segment_image(segmentation_model, image1, viz_path_masking_1, args.device)
                    subfolder_mask[f"{subfolder}_{pair_folder_name}"] = (mask0, mask1)
                    print(f"{model_name}, {subfolder}, [{pair_folder_name}]: {mask0.shape}, {mask1.shape}, {image0.dtype}, {image1.dtype}")

                #  If 'gt' has been already processed apply the saved mask to the other model folders
                if(gt_flag==True and model_name != gt_folder):
                    print(f"Taking mask from {subfolder} and {pair_folder_name}")
                    mask0 = subfolder_mask[f"{subfolder}_{pair_folder_name}"][0]
                    mask1 = subfolder_mask[f"{subfolder}_{pair_folder_name}"][1]
                    viz_path_masking_0 = os.path.join(curr_path_folder_image, f"output_{subfolder}_{image_0_name}_{img_1_name}_mask_0.jpg")
                    viz_path_masking_1 = os.path.join(curr_path_folder_image, f"output_{subfolder}_{image_0_name}_{img_1_name}_mask_1.jpg")
                    overlay_mask(image0, mask0, viz_path_masking_0)
                    overlay_mask(image1, mask1, viz_path_masking_1)
                    print(f"{model_name}, {subfolder}, [{pair_folder_name}]: {mask0.shape}, {mask1.shape}, {image0.shape}, {image1.shape}")

                #  If 'gt' is not present, apply the mask indipendetly for each model folder
                elif(gt_flag==False and model_name != gt_folder):
                    mask0 = segment_image(segmentation_model, image0, viz_path_masking_0, args.device)
                    mask1 = segment_image(segmentation_model, image1, viz_path_masking_1, args.device)                    

                # Perform keypoint matching between the two images
                result = matcher(image0, image1)

                # Filter the matched keypoints based on the masks (using the mask_number for the specified class)
                filtered_result = masking_result(result, mask_number, mask0, mask1)

                # Print the paths and the number of inliers found after RANSAC
                out_str = f"Paths: {str(img0_path), str(img1_path)}. \n Found {filtered_result['num_inliers']} inliers after RANSAC. "

                # If visualization is enabled and there are inliers, save the visualization
                if not args.no_viz and filtered_result["num_inliers"] != 0:
                    plot_matches(image0, image1, filtered_result, save_path=viz_path_matching)
                    out_str += f"Viz saved in {viz_path_matching}. "

                # Add additional information to the filtered result (e.g., paths, matcher type, image size)
                filtered_result["img0_path"] = img0_path
                filtered_result["img1_path"] = img1_path
                filtered_result["matcher"] = args.matcher
                filtered_result["n_kpts"] = args.n_kpts
                filtered_result["im_size"] = args.im_size

                # Save the filtered result as a Torch file in the output directory
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
    parser.add_argument("--im_size", type=int, default=1024, help="resize img to im_size x im_size")
    parser.add_argument("--n_kpts", type=int, default=2048, help="max num keypoints")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--no_viz", action="store_true", help="avoid saving visualizations")
    parser.add_argument("--input",type=Path,default=None,help="path to either (1) dir with dirs with image pairs or (2) txt file with two image paths per line")
    parser.add_argument("--out_dir", type=Path, default=None, help="path where outputs are saved") # frames_matched\name_folder_matched 
    parser.add_argument("--analysis",action="store_true", help="making the analysis of robustness without the inference process") 
    parser.add_argument("--extract_keypoints", action="store_true", help="making the analysis of robustness without the inference process") 
    parser.add_argument("--mask_type", type=int, default=None)
    parser.add_argument("--gt_folder", type=str, default=None, help="path to the gt folder that is used for extracting the mask that will beused for the otehr model folders") 

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"outputs_{args.matcher}")

    return args


def main(args):

    assert args.mask_type in range(1,21) or args.mask_type is None

    if(args.analysis):
        assert not(args.out_dir is None)
        robustness_analysis(args.out_dir)

    if(args.extract_keypoints):
        #args in to consider the input folder, args out to write the ouput
        assert not(args.input is None and args.out is None)
        seg_model = load_segmentation_model(device=args.device)
        
        with open('pascalVOC.json') as f:
            if(args.mask_type in range(1,21)):
                VOC_classes = json.load(f)
                class_chosen = VOC_classes[str(args.mask_type)]
                print(f"Class to be masked: {class_chosen}")
            f.close()

        extract_keypoints(args, args.mask_type, seg_model, args.gt_folder)

        

if __name__ == "__main__":
    args = parse_args()
    main(args)

    
    
    

