# 🎞️ Video Frame Pairing & Semantic Matching Tool
A Python tool for manually extracting and organizing frame pairs from videos, designed for use in tasks like image matching, super-resolution, and temporal consistency training.

✨ The tool also supports a post-processing phase for keypoint extraction and matching, where keypoints are filtered by semantic segmentation masks (e.g., only on objects like boats, people, cars). This enables precise, object-aware evaluation and dataset creation for vision-based models.

Whether you're building a training dataset, analyzing model robustness, or benchmarking matching algorithms — this tool streamlines the process from video to clean, structured pairs with deep matching capabilities.

---

## 📌 Features

- 🖼️ **Manual Frame Pairing**  
  Easily browse videos and select frame pairs with simple keyboard shortcuts. Ideal for creating datasets for super-resolution, image matching, or temporal consistency.

- ⏩ **Flexible Frame Skipping**  
  Quickly navigate through the video using short (`c`) and long (`x`) skip intervals to efficiently reach points of interest.

- 🧼 **Automatic Border Removal**  
  Optionally detect and remove black borders from each frame to improve alignment and visual consistency.

- 🗂️ **Structured Output Organization**  
  Automatically saves and organizes extracted frame pairs into nested folders based on video name and pair ID, simplifying data management and labeling.

- 🔍 **Segmentation-Aware Keypoint Matching** *(NEW)*  
  Run a second-stage process to extract and match keypoints **only within semantically segmented regions** (e.g., only on boats or people) using DeepLabV3. Ensures precise matching on objects of interest.

- 🔁 **Shared Ground Truth Masks Support**  
  Extract masks from a `gt` folder and reuse them across other models (e.g., `lq`, `hat`) to ensure consistent object-based filtering and fair comparison.


---

## 📸 Demo

| Current Frame | Future Frame |
|---------------|--------------|
|  <img src="frames_paired\salient_frames_855189-hd_1920_1080_30fps\gt\pair_0\855189-hd_1920_1080_30fps_pair_0_1.jpg" /> | <img src="frames_paired\salient_frames_855189-hd_1920_1080_30fps\gt\pair_0\855189-hd_1920_1080_30fps_pair_0_2.jpg" />|

| Frames not masked and matched|
|---------------|
|  <img src="frames_matched_wihtout_mask\salient_frames_855189-hd_1920_1080_30fps\gt\output_gt_855189-hd_1920_1080_30fps_pair_0_1_855189-hd_1920_1080_30fps_pair_0_2_matches.jpg" />|

| Frames masked and matched|
|---------------|
|  <img src="frames_matched_wiht_mask\salient_frames_855189-hd_1920_1080_30fps\gt\output_gt_855189-hd_1920_1080_30fps_pair_0_1_855189-hd_1920_1080_30fps_pair_0_2_matches.jpg" />|

---

## 🧠 How It Works

1. **Video Browsing**:
   - The user is shown a current frame and another one a few frames ahead (configurable).
   - Press:
     - `s`: Save the current pair.
     - `x`: Skip many frames.
     - `c`: Skip few frames.
     - `z`: Exit the video.
   - Pairs are saved as `video_name_pair_N_1.jpg` and `video_name_pair_N_2.jpg`.

2. **Automatic Folder Organization**:
   - After frame selection, the script automatically scans the output directory, detects saved pairs, and organizes them into subfolders.


3. **💡 Keypoint Matching with Segmentation-based Masking [NEW]**:

  - You can run a post-processing phase to extract and match keypoints only in semantically relevant areas of the images (e.g., only boats, people, etc.) using a pretrained segmentation model.

  - The DeepLabV3 segmentation model is applied on each frame.

  - Keypoints are matched only within a specific object class (defined by a Pascal VOC class ID).



--- 
## 🆕 Semantic-Aware Keypoint Matching
### - 📥 Step 1: Save Frame Pairs
First, use the basic tool to extract and save frame pairs from your video(s).

```bash
python collecting_salient_frames.py \
    --input /path/to/video_or_folder \
    --out_dir output_folder \
    --remove_contours True \
    --inner_folder_name frames \
    --frames_difference 1 \
    --skip_interval_long 20 \
    --skip_interval_small 3 \
    --images_to_be_paired True
```
- Suppose you processed videos from videos/ and saved frames in output_folder/, the structure might look like:
```
output_folder/
├── salient_name_video/
│   └── inner_folder_name/
│       └── pair0 
│          ├── video1_pair_0_1.jpg
│          ├── video1_pair_0_2.jpg
│       └── pair 1
│          ├── ...

```

### - 🧠 Step 2: Extract Keypoints within Specific Masked Regions
Once pairs are extracted, run the following:

```bash
python main_matcher.py \
    --extract_keypoints \
    --input output_folder \
    --out_dir frames_matched \
    --mask_type 4 \
    --matcher sift-lg \
```
- 🧠 Smart Mask Sharing via "--gt_folder"
  
If a --gt_folder is provided, the tool uses the segmentation masks generated from that folder and reuses them for matching in all other folders (e.g., different models or augmentations of the same video).

🔁 This ensures that all matchings across models (e.g., gt, hat, lq) are filtered using consistent masks, enabling robust comparisons and fair benchmarking.

📌 Example:
--input frames_paired/salient_video/
```bash
├── gt/           ← masks are extracted here
├── hat/          ← same images, different model
├── lq/           ← low-quality version
```
```bash
--gt_folder gt 
--mask_type 4
```
### 🔧 Parameters

| Argument            | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `--extract_keypoints` | Enables keypoint extraction and matching                                   |
| `--mask_type`         | Pascal VOC class ID to filter keypoints (e.g., `4` = boat, `15` = person) |
| `--gt_folder`         | Folder used to extract segmentation masks and apply them to other folders |
| `--matcher`           | Keypoint matcher to use (`sift-lg`, `superpoint-lightglue`, etc.)         |

---
### Acknowledgements
  
Special thanks to the authors of the respective works that are included in this repo (see their papers above). Additional thanks to [@GrumpyZhou](https://github.com/GrumpyZhou) for developing and maintaining the [Image Matching Toolbox](https://github.com/GrumpyZhou/image-matching-toolbox/tree/main), which we have wrapped in this repo, and the [maintainers](https://github.com/kornia/kornia?tab=readme-ov-file#community) of [Kornia](https://github.com/kornia/kornia).


## Cite
This repo was created as part of the EarthMatch paper. Please consider citing EarthMatch if this repo is helpful to you!

```
@InProceedings{Berton_2024_EarthMatch,
    author    = {Berton, Gabriele and Goletto, Gabriele and Trivigno, Gabriele and Stoken, Alex and Caputo, Barbara and Masone, Carlo},
    title     = {EarthMatch: Iterative Coregistration for Fine-grained Localization of Astronaut Photography},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
}
```
