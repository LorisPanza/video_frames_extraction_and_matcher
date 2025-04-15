# 🎞️ Video Frame Pairing Tool

This Python utility helps you **manually extract and save frame pairs** from videos for use in tasks such as **image matching**, **super-resolution**, or **temporal consistency training**. It also provides a post-processing function to **automatically organize the saved pairs** into structured folders.

---

## 📌 Features

- 🚀 Manually browse through video frames.
- 🖼️ Save pairs of visually similar frames by pressing a key.
- 🧼 Optional automatic removal of black borders from each frame.
- ⏩ Skip frames quickly (long and short intervals).
- 🗂️ Automatically organize saved frames into structured folders based on their naming pattern.

---

## 📸 Demo

| Current Frame | Future Frame |
|---------------|--------------|
|  <img src="frames_paired\salient_frames_855189-hd_1920_1080_30fps\gt\pair_0\855189-hd_1920_1080_30fps_pair_0_1.jpg" /> | <img src="frames_paired\salient_frames_855189-hd_1920_1080_30fps\gt\pair_0\855189-hd_1920_1080_30fps_pair_0_2.jpg" />|

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
   - After frame selection, the script automatically scans the output directory, detects saved pairs, and organizes them into subfolders named after the prefix (`video_name`) of the images.


3. **💡 Keypoint Matching with Segmentation-based Masking [NEW]**:

  - You can run a post-processing phase to extract and match keypoints only in semantically relevant areas of the images (e.g., only boats, people, etc.) using a pretrained segmentation model.

  - The DeepLabV3 segmentation model is applied on each frame.

  - Keypoints are matched only within a specific object class (defined by a Pascal VOC class ID).



--- 
## 🆕 Semantic-Aware Keypoint Matching
- 📥 Step 1: Save Frame Pairs
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
### 🗂️ Folder structure
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

- 🧠 Step 2: Extract Keypoints within Specific Masked Regions
Once pairs are extracted, run the following:

```bash
python main.py \
    --extract_keypoints \
    --input output_folder \
    --out_dir frames_matched \
    --mask_type 4 \
    --matcher sift-lg \
```
---

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
