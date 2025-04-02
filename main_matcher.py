"""
This script performs image matching using a specified matcher model. It processes pairs of input images,
detects keypoints, matches them, and performs RANSAC to find inliers. The results, including visualizations
and metadata, are saved to the specified output directory.
"""

import torch
import argparse
from pathlib import Path
import os
import numpy as np
import re

<<<<<<< HEAD
from matching.utils import get_image_pairs_paths, get_default_device
=======
from matching.utils import get_image_pairs_paths, get_model_folders, load_torch_save, create_images_folder
>>>>>>> 6aa7887 (introducing salient frames extractor and code to compare the pairs and extract statistics)
from matching import get_matcher, available_models
from matching.viz import plot_matches, plot_barplot_curr_folder


<<<<<<< HEAD
=======
# This is to be able to use matplotlib also without a GUI
#if not hasattr(sys, "ps1"):
#    matplotlib.use("Agg")

>>>>>>> 6aa7887 (introducing salient frames extractor and code to compare the pairs and extract statistics)

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
def robustness_analysis(out_dir, plot = False):
    model_output_folders = get_model_folders(out_dir) 
    dict_result_analysis = {}
    pair_pattern = re.compile(r"(pair_\d+)")
    dict_performances = {}
    performances_order = []

    for out_folder in model_output_folders:
        performances_order.append(out_folder)
        folder_image_names = Path(out_folder).glob("*")
        for folder_image in folder_image_names:
            out_folder_name = os.path.split(folder_image)[-1]
            results_curr_folder = load_torch_save(folder_image)
            out_str = f"Extracted {len(results_curr_folder)} results from {out_folder_name}."
            print(out_str)

            ratio_arr = []
            curr_images_name = []
            if out_folder_name not in dict_performances.keys():
                print(f"Initializing {out_folder_name}")
                dict_performances[out_folder_name] = []

            for str_res in results_curr_folder:
                #loading torch file
                res = torch.load(str_res)
                assert len(res["matched_kpts1"]) == len(res["matched_kpts0"])

                # metric
                curr_ratio = res["num_inliers"]/len(res["matched_kpts0"]) #it should be equal to 1
                ratio_arr.append(curr_ratio)

                # name
                match = pair_pattern.search(str_res) #Path(str_res).stem
                imgs_name = match.group(1)
                curr_images_name.append(imgs_name)

            #mean and plot
            #TODO: Techniques to visualize the metrics
            mean_ratio_arr = round(np.mean(ratio_arr),2)
            dict_performances[out_folder_name].append(mean_ratio_arr)
            print(f"Mean ratio {curr_images_name}:{ratio_arr}")
            if plot:
                plot_barplot_curr_folder(ratio_arr, curr_images_name, out_folder_name)
            out_str = f"{out_folder_name} ratio: {mean_ratio_arr:0.2f}."
            print(out_str)
            dict_result_analysis[out_folder_name] = mean_ratio_arr
    
    print(f"Order of performances dict: {performances_order}")
    for key,val in dict_performances.items():
        print(f"Mean array {key}: {val}")



def sr_robustness_data(args):
    image_size = [args.im_size, args.im_size]
    args.out_dir.mkdir(exist_ok=True, parents=True)
    
    # Choose a matcher
    matcher = get_matcher(args.matcher, device=args.device, max_num_keypoints=args.n_kpts)
    model_folders = get_model_folders(args.input)
    for model in model_folders:

        # output/model creation
        model_name = os.path.split(model)[1]
        out_model_path = os.path.join(args.out_dir,model_name)
        os.makedirs(name=out_model_path, exist_ok=True)

        # input_dir/model/image_folders
        subfolders = os.listdir(model)

        for subfolder in subfolders:
            subfolder_path = os.path.join(model, subfolder)
            pairs_of_paths, folder_names = get_image_pairs_paths(subfolder_path)

            # input_dir/model/image_folders/pair_folder/...png
            for i, (img0_path, img1_path) in enumerate(pairs_of_paths):
                image_0_name = Path(img0_path).stem
                img_1_name = Path(img1_path).stem
                image0 = matcher.load_image(img0_path, resize=image_size)
                image1 = matcher.load_image(img1_path, resize=image_size)
                result = matcher(image0, image1)

                out_str = f"Paths: {str(img0_path), str(img1_path)}. Found {result['num_inliers']} inliers after RANSAC. "
                #curr_folder_name = os.path.split(folder_names[i])[-1]

                #output_dir/model/image_folders/   creation
                curr_path_folder_image = os.path.join(out_model_path, subfolder)
                os.makedirs(name=curr_path_folder_image, exist_ok=True)

                if not args.no_viz: 
                    #output_dir/model/image_folders/result.png-torch
                    viz_path = os.path.join(curr_path_folder_image, f"output_{subfolder}_{image_0_name}_{img_1_name}_matches.jpg") #str(out_model_path) / f"output_{i}_matches.jpg"
                    plot_matches(image0, image1, result, save_path=viz_path)
                    out_str += f"Viz saved in {viz_path}. "

                result["img0_path"] = img0_path
                result["img1_path"] = img1_path
                result["matcher"] = args.matcher
                result["n_kpts"] = args.n_kpts
                result["im_size"] = args.im_size

                dict_path = os.path.join(curr_path_folder_image, f"output_{subfolder}_{image_0_name}_{img_1_name}_result.torch")
                torch.save(result, dict_path)
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
    parser.add_argument("--device", type=str, default=get_default_device(), choices=["cpu", "cuda"])
    parser.add_argument("--no_viz", action="store_true", help="avoid saving visualizations")

    parser.add_argument(
        "--input",
<<<<<<< HEAD
        type=Path,
        nargs="+",  # Accept one or more arguments
        default=[Path("assets/example_pairs")],
        help="path to either (1) two image paths or (2) dir with two images or (3) dir with dirs with image pairs or "
        "(4) txt file with two image paths per line",
=======
        type=str,
        default=None, #"assets/example_pairs"
        help="path to either (1) dir with dirs with image pairs or (2) txt file with two image paths per line",
>>>>>>> 6aa7887 (introducing salient frames extractor and code to compare the pairs and extract statistics)
    )
    parser.add_argument("--out_dir", type=Path, default=None, help="path where outputs are saved") # it generates an output path like: output_dir/models/image_folder/results.png/torch
    #parser.add_argument("--sr_robustness", action="store_true", help="Apply the matcher and make the comparison and create a metric among swin, hat and pipeline output to prove the robustness of the model.") # it takes as input a path like: input_dir/models/image_folder/pair_folder/image.png
    parser.add_argument("--only_analysis", action="store_true", help="making the analysis of robustness without the inference process") # it takes as input a path like output_dir/models/image_folder/results.png/torch and make an analysis over the avg value of matching among the pairs for gt,hat,...
    parser.add_argument("--images_to_be_paired", type=Path, default=None, help ="pair the image in folders by the name.") 

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"outputs_{args.matcher}")

    return args


if __name__ == "__main__":
    args = parse_args()

    print(args)

    if(not(args.images_to_be_paired is None)):
        print("Creating folder")
        # args.out to save matched frames
        assert not(args.out_dir is None)
        create_images_folder(args.images_to_be_paired, args.out_dir)
    
    if(args.only_analysis):
        assert not(args.out_dir is None)
        robustness_analysis(args.out_dir)
    else:
        #args in to consider the input folder, args out to write the ouput
        assert not(args.input is None and args.out is None)
        sr_robustness_data(args)
    
    

