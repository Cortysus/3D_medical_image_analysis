import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from glob import glob
import nibabel as nib


class NiftiDataset(Dataset):
    def __init__(self, source_dir, target_dir, transforms=None):
        """Create a dataset class in PyTorch for reading NIfTI files

        Args:
            source_dir (str): input directory
            target_dir (str): output directory
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.source_fns = glob(os.path.join(source_dir, "*.nii*"))
        self.target_fns = glob(os.path.join(target_dir, "*.nii*"))
        assert (
            len(self.source_fns) == len(self.target_fns) and len(self.target_fns) != 0
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.source_fns)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_source = nib.load(self.source_fns[idx]).get_fdata()
        image_target = nib.load(self.target_fns[idx]).get_fdata()
        sample = {"source": image_source, "target": image_target}

        if self.transforms:
            sample = self.transforms(sample)

        return sample


class RandomCrop3D(object):
    """Crop randomly the 3D image in a sample.

    Args:
        output_size (tuple or int): desired output size. If int, cube is made
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        source, target = sample["source"], sample["target"]
        h, w, d = source.shape[:3]
        new_h, new_w, new_d = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        down = np.random.randint(0, d - new_d)

        source = source[top : top + new_h, left : left + new_w, down : down + new_d]
        target = target[top : top + new_h, left : left + new_w, down : down + new_d]

        return {"source": source, "target": target}
