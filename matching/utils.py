import logging
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as tfm
import os, contextlib
from yacs.config import CfgNode as CN
import sys
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import shutil, re



logger = logging.getLogger()
logger.setLevel(31)  # Avoid printing useless low-level logs


def get_image_pairs_paths(inputs):

    pair_dirs = sorted(Path(inputs).glob("*"))
    #print(pair_dirs) # first level (3 trees, BOSS)
    pairs_of_paths = [list(pair_dir.glob("*")) for pair_dir in pair_dirs]
    #print(pairs_of_paths) # second level (list of 2 images for each folder)
    for pair in pairs_of_paths:
        if len(pair) != 2:
            raise RuntimeError(f"{pair} should be a pair of paths")
    return pairs_of_paths, pair_dirs


def get_model_folders(inputs):
    inputs = Path(inputs)
    if not inputs.exists():
        raise RuntimeError(f"{inputs} does not exist")
    dirs = sorted(Path(inputs).glob("*"))
    return dirs

def create_paired_folder(output_model):
    """given a folder with images, create subfolders and move files into them based on pairs

    Args:
        sorce_path (torch.Tensor | np.ndarray | dict | list): the path should be outer_folder/models/list_of_images.png or outer_folder/models/image_folders/list_of_images.png

    Returns:
        None
    """
    pair_pattern = re.compile(r"(pair_\d+)")
    for folder_image in sorted(Path(output_model).glob("*")):
        # Create subfolders and move files into them based on pairs
        files = [f for f in os.listdir(folder_image) if f.endswith(".jpg")]
        for file in files:
            match = pair_pattern.search(file)  # Search for the pair identifier
            if match:
                pair_id = match.group(1)  # Extract the pair identifier (e.g., "pair_n")
                subfolder = os.path.join(folder_image, pair_id)
            # Create subfolder if it doesn't exist
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            # Move the file into the corresponding subfolder
            shutil.move(os.path.join(folder_image, file), os.path.join(subfolder, file))


def pair_images_in_folder(source_path, dest_path):
    """pair the image having pair_n_0 and pair_n_1 name

    Args:
        sorce_path (torch.Tensor | np.ndarray | dict | list): the path should be source_path/models/list_of_images.png or outer_folder/models/image_folders/list_of_images.png

    Returns:
        None
    """

    inputs = Path(source_path)
    if not inputs.exists():
        raise RuntimeError(f"{inputs} does not exist")
    
    # dirs containt the models name
    dirs = sorted(Path(inputs).glob("*"))

    # for each single model (hat, pipeline, gt)
    for dir in dirs:
        print(f"Actual dir: {dir}")
        # input_dir/model
        model_name = os.path.split(dir)[1]
        model_folder = os.path.join(source_path, model_name)

        # create output folder structure having same name
        output_folder_model = os.path.join(dest_path,model_name)
        
        # Case 1: input/models/list_of_images.png
        images = [f for f in os.listdir(model_folder) if f.endswith(".jpg")]

        start_pos_folder = 0

        if(len(images)) != 0:
            for img in images:

                path_image = os.path.join(dir, img)
                end_pos_folder = img.find("_pair")

                folder_name = img[start_pos_folder:end_pos_folder]

                #input_dir/model/folder_image
                output_folder_name = os.path.join(output_folder_model, folder_name)

                os.makedirs(name = output_folder_name, exist_ok = True)

                #input_dir/model/folder_image/image.png
                shutil.copy(path_image, output_folder_name+"/"+img)

        # Case 2: outer_folder/models/image_folders/list_of_images.png
        else:
            #print("Path be like: Outer_folder/models/image_folders/list_of_images.png")
            folders = [f for f in os.listdir(model_folder)]
            for folder in folders:
                path_folder = os.path.join(dir,folder)
                output_folder_name = os.path.join(output_folder_model, folder)
                os.makedirs(name = output_folder_name, exist_ok = True)
                shutil.copytree(path_folder, output_folder_name , dirs_exist_ok=True)
        
        create_paired_folder(output_folder_model)


def load_torch_save(inputs):
    torch_files = [f for f in os.listdir(inputs) if f.endswith('.torch')]
    torch_files_path = [os.path.join(inputs, f) for f in torch_files]
    return torch_files_path


def to_numpy(x: torch.Tensor | np.ndarray | dict | list) -> np.ndarray:
    """convert item or container of items to numpy

    Args:
        x (torch.Tensor | np.ndarray | dict | list): input

    Returns:
        np.ndarray: numpy array of input
    """
    if isinstance(x, list):
        return np.array([to_numpy(i) for i in x])
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = to_numpy(v)
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x


def to_tensor(x: np.ndarray | torch.Tensor, device: str = None) -> torch.Tensor:
    """Convert to tensor and place on device

    Args:
        x (np.ndarray | torch.Tensor): item to convert to tensor
        device (str, optional): device to place tensor on. Defaults to None.

    Returns:
        torch.Tensor: tensor with data from `x` on device `device`
    """
    if isinstance(x, torch.Tensor):
        pass
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if device is not None:
        return x.to(device)


def to_normalized_coords(pts: np.ndarray | torch.Tensor, height: int, width: int):
    """normalize kpt coords from px space to [0,1]
    Assumes pts are in x, y order in array/tensor shape (N, 2)

    Args:
        pts (np.ndarray | torch.Tensor): array of kpts, must be shape (N, 2)
        height (int): height of img
        width (int): width of img

    Returns:
        np.array: kpts in normalized [0,1] coords
    """
    # normalize kpt coords from px space to [0,1]
    # assume pts are in x,y order
    assert pts.shape[-1] == 2, f"input to `to_normalized_coords` should be shape (N, 2), input is shape {pts.shape}"
    pts = to_numpy(pts).astype(float)
    pts[:, 0] /= width
    pts[:, 1] /= height

    return pts


def to_px_coords(pts: np.ndarray | torch.Tensor, height: int, width: int) -> np.ndarray:
    """unnormalized kpt coords from [0,1] to px space
    Assumes pts are in x, y order

    Args:
        pts (np.ndarray | torch.Tensor): array of kpts, must be shape (N, 2)
        height (int): height of img
        width (int): width of img

    Returns:
        np.array: kpts in normalized [0,1] coords
    """
    assert pts.shape[-1] == 2, f"input to `to_px_coords` should be shape (N, 2), input is shape {pts.shape}"
    pts = to_numpy(pts)
    pts[:, 0] *= width
    pts[:, 1] *= height

    return pts


def resize_to_divisible(img: torch.Tensor, divisible_by: int = 14) -> torch.Tensor:
    """Resize to be divisible by a factor. Useful for ViT based models.

    Args:
        img (torch.Tensor): img as tensor, in (*, H, W) order
        divisible_by (int, optional): factor to make sure img is divisible by. Defaults to 14.

    Returns:
        torch.Tensor: img tensor with divisible shape
    """
    h, w = img.shape[-2:]

    divisible_h = round(h / divisible_by) * divisible_by
    divisible_w = round(w / divisible_by) * divisible_by
    img = tfm.functional.resize(img, [divisible_h, divisible_w], antialias=True)

    return img


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)

    return wrapper


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def load_module(module_name: str, module_path: Path | str) -> None:
    """Load module from `module_path` into the interpreter with the namespace given by module_name.

    Note that `module_path` is usually the path to an `__init__.py` file.

    Args:
        module_name (str): module name (will be used to import from later, as in `from module_name import my_function`)
        module_path (Path | str): path to module (usually an __init__.py file)
    """
    import importlib

    # load gluefactory into namespace
    # module_name = 'gluefactory'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def add_to_path(path: str | Path, insert=None) -> None:
    path = str(path)
    if path in sys.path:
        sys.path.remove(path)
    if insert is None:
        sys.path.append(path)
    else:
        sys.path.insert(insert, path)

def get_default_device():
    device = "cpu"

    if sys.platform == "darwin" and torch.backends.mps.is_available():
        device = "mps"

    elif torch.cuda.is_available():
        device = "cuda"

    return device
