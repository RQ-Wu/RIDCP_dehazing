import os
import cv2
import random
import numpy as np
from torch.utils import data as data
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

from .data_util import make_dataset

def uint2single(img):
    return np.float32(img/255.)

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:   # add color Gaussian noise
        img += np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4: # add grayscale Gaussian noise
        img += np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:            # add  noise
        L = noise_level2/255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img

def add_JPEG_noise(img):
    quality_factor = random.randint(30, 95)
    img = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)
    return img

@DATASET_REGISTRY.register()
class HazeOnlineDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(HazeOnlineDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.depth_folder = opt['dataroot_depth']
        self.gt_paths = make_dataset(self.gt_folder)
        self.depth_paths = make_dataset(self.depth_folder)
        self.beta_range = opt['beta_range']
        self.A_range = opt['A_range']
        self.color_p = opt['color_p']
        self.color_range = opt['color_range']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        #  scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        img_gt = cv2.imread(gt_path).astype(np.float32) / 255.0

        depth_path = os.path.join(self.depth_folder, gt_path.split('/')[-1].split('.')[0] + '.npy')
        img_depth = np.load(depth_path)
        img_depth = (img_depth - img_depth.min()) / (img_depth.max() - img_depth.min())

        beta = np.random.rand(1) * (self.beta_range[1] - self.beta_range[0]) + self.beta_range[0]
        t = np.exp(-(1- img_depth) * 2.0 * beta)
        t = t[:, :, np.newaxis]
        
        A = np.random.rand(1) * (self.A_range[1] - self.A_range[0]) + self.A_range[0]
        if np.random.rand(1) < self.color_p:
            A_random = np.random.rand(3) * (self.color_range[1] - self.color_range[0]) + self.color_range[0]
            A = A + A_random
        
        
        img_lq = img_gt.copy()
        # adjust luminance
        if np.random.rand(1) < 0.5:
            img_lq = np.power(img_lq, np.random.rand(1) * 1.5 + 1.5)
        # add gaussian noise
        if np.random.rand(1) < 0.5:
            img_lq = add_Gaussian_noise(img_lq)

        # add haze
        img_lq = img_lq * t + A * (1 - t)

        # add JPEG noise
        if np.random.rand(1) < 0.5:
            img_lq = add_JPEG_noise(img_lq)

        if img_gt.shape[-1] > 3:
            img_gt = img_gt[:, :, :3]
            img_lq = img_lq[:, :, :3]
        # augmentation for training
        if self.opt['phase'] == 'train':
            input_gt_size = np.min(img_gt.shape[:2])
            input_lq_size = np.min(img_lq.shape[:2])
            scale = input_gt_size // input_lq_size
            gt_size = self.opt['gt_size']

            if self.opt['use_resize_crop']:
                # random resize
                if input_gt_size > gt_size:
                    input_gt_random_size = random.randint(gt_size, input_gt_size)
                    input_gt_random_size = input_gt_random_size - input_gt_random_size % scale # make sure divisible by scale 
                    resize_factor = input_gt_random_size / input_gt_size
                else:
                    resize_factor = (gt_size+1) / input_gt_size
                img_gt = random_resize(img_gt, resize_factor)
                img_lq = random_resize(img_lq, resize_factor)
                t = random_resize(t, resize_factor)

                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, input_gt_size // input_lq_size,
                                               gt_path)

            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])


        if self.opt['phase'] != 'train':
            crop_eval_size = self.opt.get('crop_eval_size', None)
            if crop_eval_size:
                input_gt_size = img_gt.shape[0]
                input_lq_size = img_lq.shape[0]
                scale = input_gt_size // input_lq_size
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, crop_eval_size, input_gt_size // input_lq_size,
                                               gt_path)

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.gt_paths)
