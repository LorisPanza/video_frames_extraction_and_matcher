# 🎞️ Video Frame Pair Extraction Tool

> A handy Python tool for extracting **salient frame pairs** from videos, with manual selection and border removal features. Perfect for training data creation in tasks like **frame matching**, **super-resolution**, and **temporal alignment**.

---

## 🚀 Features

- ✅ **Interactive frame selection**: View and save meaningful pairs with keyboard controls.
- ✂️ **Automatic black border removal**.
- ⏩ **Frame skipping**: Skip frames quickly with custom intervals.
- 🎥 Supports `.avi` and `.mp4` video formats.
- 📁 Automatically organizes output in structured folders.

---

## 📸 Demo

| Current Frame | Future Frame |
|---------------|--------------|
|  <img src="assets\frames_extracted\salient_frames_855189-hd_1920_1080_30fps\gt\855189-hd_1920_1080_30fps_pair_0_1.jpg" /> | <img src="assets\frames_extracted\salient_frames_855189-hd_1920_1080_30fps\gt\855189-hd_1920_1080_30fps_pair_0_1.jpg" />|


---

## 🧠 How It Works

The script reads a video, skips forward by a user-defined number of frames (`frames_difference`), and allows you to decide if the current frame and the future one form a useful pair. You can then:

- Press `s` to **save the frame pair**.
- Press `x` to **skip many frames** (long jump).
- Press `c` to **skip a few frames** (short jump).
- Press `z` to **stop the session**.

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
