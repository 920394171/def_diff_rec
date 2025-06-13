import os

import h5py
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import nibabel as nib
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.ndimage import zoom
import numpy as np
# import SimpleITK as sitk

# print(os.getcwd())
import sys

sys.path.append('./')
from Dataloader.dataloader_utils import *

EPS = 1e-7


def get_dataloader(data_name='cmr', mode='train'):
    if data_name == 'cmr':
        if mode == 'train':
            dataloader = CMR_loader
        elif mode == 'aug':
            dataloader = CMR_tgt_loader
        else:
            print('mode not exist')
    elif data_name == 'lct':
        if mode == 'train':
            dataloader = LCT_loader
        elif mode == 'aug':
            dataloader = LCT_tgt_loader
        else:
            print('mode not exist')
    elif data_name == 'synapse':
        if mode == 'train':
            dataloader = Synapse_loader
        elif mode == 'aug':
            dataloader = Synapse_loader
        else:
            print('mode not exist')

    else:
        print('dataloader not exist')
    return dataloader


class LCT_loader(Dataset):
    def __init__(self, data_root_path=f'Data/Src_data/CTLung_processed/', target_res=(256, 256), transforms=None,
                 noise_scale=0.0, patient_index=None):
        # def __init__(self, data_root_path = '/home/data/jzheng/CTLung_processed/', target_res = (256, 256),transforms = None, noise_scale=0.0, patient_index = None):
        self.files = [data_root_path + f for f in os.listdir(data_root_path) if f.endswith('.npy')]
        self.transforms = transforms
        self.noise_scale = noise_scale
        self.d_p = data_root_path

    def __getitem__(self, item):
        array = np.load(self.files[item])
        if 'process' not in self.d_p:
            array = (array - array.min()) / (array.max() - array.min() + EPS)  # Normalize to 0 to 1
        array = array[None, :, :, :]  # add a channel to array make it (‘C’,H,W,Z)
        if self.transforms != None:
            array = self.transforms(array)
        # print(array.shape)
        return array, array, item  # -> (B, C, H, W, Z)
        # return array, array # -> (B, C, H, W, Z)

    def __len__(self):
        return len(self.files)


class LCT_tgt_loader(Dataset):
    def __init__(self, data_root_path="Data/Tgt_data/lct/", noise_scale=0.0, patient_index=None):
        self.files_gt = [data_root_path + "Gt/" + f for f in os.listdir(data_root_path + "Gt/")]
        self.files_tr = [data_root_path + 'Tr/' + f for f in os.listdir(data_root_path + "Tr/")]

        self.files_tr.sort()
        self.files_gt.sort()

        self.transforms = transforms
        self.noise_scale = noise_scale

    def __getitem__(self, item):
        img_nib = nib.load(self.files_tr[item])
        mask_nib = nib.load(self.files_gt[item])

        image = img_nib.get_fdata()
        mask = mask_nib.get_fdata()

        image = image[None, :, :, :]
        mask = mask[None, :, :, :]

        print(self.files_tr[item], self.files_gt[item])

        return image, mask, item

    def __len__(self):
        assert len(self.files_gt) == len(self.files_tr)
        return len(self.files_gt)


class LCT_seg(Dataset):
    def __init__(self, data_root_path="/home/data/jzheng/CTLung_processed/testset/modality_0001/", noise_scale=0.0,
                 patient_index=None):
        self.files_gt = [data_root_path + "Gt/" + f for f in os.listdir(data_root_path + "Gt/")]
        self.files_tr = [data_root_path + 'Tr/' + f for f in os.listdir(data_root_path + "Tr/")]

        self.files_tr.sort()
        self.files_gt.sort()

        self.transforms = transforms
        self.noise_scale = noise_scale

    def __getitem__(self, item):
        img_nib = nib.load(self.files_tr[item])
        mask_nib = nib.load(self.files_gt[item])

        image = img_nib.get_fdata()
        mask = mask_nib.get_fdata()

        image = image[None, :, :, :]
        mask = mask[None, :, :, :]

        print(self.files_tr[item], self.files_gt[item])

        return image, mask, item

    def __len__(self):
        assert len(self.files_gt) == len(self.files_tr)
        return len(self.files_gt)


class Synapse_loader(Dataset):
    def __init__(self,
                 data_root_path,  # e.g. "/data/ssd1/ttzz/dataset/Synapse_h5"
                 labeled_list,  # e.g. ['0002','0023','0034','0039']
                 unlabeled_list,  # e.g. ['0001','0003',...,'0038']
                 transform=None):
        self.base_dir = data_root_path.rstrip('/')
        self.transform = transform

        all_keys = labeled_list + unlabeled_list
        self.files = [
            os.path.join(self.base_dir, f"{key}.h5")
            for key in all_keys
        ]

        if len(self.files) == 0:
            raise RuntimeError(f"No H5 files found in {self.base_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        h5_path = self.files[idx]
        with h5py.File(h5_path, 'r') as h5f:
            image = h5f['image'][:]  # e.g. (C, H, W, [D])
            label = h5f['label'][:]  # e.g. (C, H, W, [D])

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        image = sample['image']

        return image, image, idx


class CMR_loader_preprocess(Dataset):
    # This is for pre_processing for CMR. not use for training model
    def __init__(self, data_path='Data/CTLung_processed/', target_res=(256, 256), transforms=None, noise_scale=0.0):
        # def __init__(self, data_path = '/home/data/jzheng/CMR_processed/', target_res = (256, 256), transforms = None, noise_scale=0.0):
        self.d_p = data_path
        self.target_res = target_res
        self.files = [self.d_p + x for x in os.listdir(self.d_p)]
        self.transforms = transforms
        self.noise_scale = noise_scale

    def __getitem__(self, item):
        array = nib.load(self.files[item]).get_fdata()
        array = resize(array, self.target_res, anti_aliasing=True, preserve_range=True)
        array = array[None, :, :]
        array = remove_background(array)  # jzheng 20240228
        array = (array - array.min()) / (array.max() - array.min() + EPS)

        if self.noise_scale > 0:
            array = thresh_img(array, [0, self.noise_scale])
            array = array * (np.random.normal(1, self.noise_scale * 2))

        if self.transforms != None:
            array = self.transforms(array)
        return array, self.files[item]

    def __len__(self):
        return len(self.files)


class CMR_loader(Dataset):
    #   niff format size is (H,W) for CMR
    #   CMR_processed_rmbg_resize means the niif image has been gone throught rmbg and resize offline to make trainig fast
    def __init__(self, data_path=f'Data/Src_data/CMR_processed_rmbg_resize/', target_res=(256, 256), transforms=None,
                 noise_scale=0.0):
        # def __init__(self, data_path = '/home/data/jzheng/CMR_processed_rmbg_resize/', target_res = (256, 256), transforms = None, noise_scale=0.0):
        self.d_p = data_path
        self.ndims = 2
        self.target_res = target_res
        self.files = [self.d_p + x for x in os.listdir(self.d_p)]
        self.transforms = transforms
        # self.get_transform()
        self.noise_scale = noise_scale
        self.preprocessed = 'resize' in data_path

    def __getitem__(self, item):
        array = nib.load(self.files[item]).get_fdata()
        if not self.preprocessed:
            array = resize(array, self.target_res, anti_aliasing=True, preserve_range=True)
        array = array[None, :, :]
        if not self.preprocessed:
            array = remove_background(array)  # jzheng 20240228
            array = (array - array.min()) / (array.max() - array.min() + EPS)

        # if self.noise_scale > 0:
        #   array = thresh_img(array,[0,self.noise_scale])
        #   array = array * (np.random.normal(1, self.noise_scale*2)) + np.random.normal(0, self.noise_scale*2)

        if self.transforms != None:
            array = self.transforms(array)
        return array, array, item

    def __len__(self):
        return len(self.files)

    def get_transform(self, degrees=np.pi, translate=0.125):
        # self.transforms = torchvision.transforms.RandomAffine(degrees=degrees,translate=[translate]*self.ndims,interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        self.transforms = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((hyp_parameters['img_size'], hyp_parameters['img_size'])),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomAffine(degrees=degrees, translate=[translate] * self.ndims,
                                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            # torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(0.5, 0.5)
            # Lambda(lambda x: (x - 0.5) * 2)
        ])
        return


class CMR_tgt_loader(Dataset):
    def __init__(self,
                 data_path='Data/Tgt_data/cmr/',
                 #  gt_path = '/home/data/jzheng/acdc/train_gt/',
                 target_res=(256, 256),
                 is_3d=False,
                 patient_index=[],
                 ):

        #  parameter initialize
        self.d_p = os.path.join(data_path, 'Tr', '')
        self.gt_p = os.path.join(data_path, 'Gt', '')
        self.img_files = os.listdir(self.d_p)
        self.gt_files = os.listdir(self.gt_p)
        self.p_indice = patient_index
        self.target_res_2d = target_res
        self.img_files.sort()
        self.gt_files.sort()
        self.img_samples = []
        self.gt_samples = []
        self.p_id = []

        if len(self.p_indice) == 0:
            self.p_indice = [x for x in range(1, 101)]
        # build patient-to-file correspondence
        p2f = {}
        assert len(self.gt_files) == len(self.img_files)
        print(self.p_indice)
        for i in self.p_indice:
            for gt_f, img_f in zip(self.gt_files, self.img_files):
                pf_id = gt_f.split('_')[0]
                pf_id = pf_id[-3:]
                if i == int(pf_id):
                    img_volume = nib.load(self.d_p + img_f).get_fdata()
                    gt_volume = nib.load(self.gt_p + gt_f).get_fdata()
                    assert img_volume.shape == gt_volume.shape
                    depth = img_volume.shape[2]
                    for si in range(depth):
                        img = resize(img_volume[:, :, si], self.target_res_2d, anti_aliasing=True, preserve_range=True)
                        img = (img - img.min()) / (img.max() - img.min() + EPS)

                        gt = gt_volume[:, :, si]

                        gt_1_index = gt == 1
                        gt_2_index = gt == 2
                        gt_3_index = gt == 3
                        gt_4_index = gt == 4

                        gt_1 = gt * gt_1_index
                        gt_2 = gt * gt_2_index
                        gt_3 = gt * gt_3_index
                        gt_4 = gt * gt_4_index

                        gt_1 = resize(gt_1, self.target_res_2d, anti_aliasing=True, preserve_range=True)
                        gt_2 = resize(gt_2, self.target_res_2d, anti_aliasing=True, preserve_range=True)
                        gt_3 = resize(gt_3, self.target_res_2d, anti_aliasing=True, preserve_range=True)
                        gt_4 = resize(gt_4, self.target_res_2d, anti_aliasing=True, preserve_range=True)

                        self.img_samples.append(img[np.newaxis, :, :])
                        self.gt_samples.append(np.array([gt_1, gt_2, gt_3, gt_4]))
                        self.p_id.append(i)

    def __getitem__(self, item):

        return self.img_samples[item], self.gt_samples[item], self.p_id[item]

    def __len__(self):

        assert len(self.img_samples) == len(self.gt_samples)
        return len(self.img_samples)


class acdc_seg(Dataset):
    def __init__(self,
                 data_path='/home/data/jzheng/acdc/train_images/',
                 gt_path='/home/data/jzheng/acdc/train_gt/',
                 target_res=(256, 256),
                 is_3d=False,
                 patient_index=[],
                 ):

        #  parameter initialize
        self.d_p = data_path
        self.gt_p = gt_path
        self.img_files = os.listdir(self.d_p)
        self.gt_files = os.listdir(self.gt_p)
        self.p_indice = patient_index
        self.target_res_2d = target_res
        self.img_files.sort()
        self.gt_files.sort()
        self.img_samples = []
        self.gt_samples = []
        self.p_id = []

        if len(self.p_indice) == 0:
            self.p_indice = [x for x in range(1, 101)]
        # build patient-to-file correspondence
        p2f = {}
        assert len(self.gt_files) == len(self.img_files)
        print(self.p_indice)
        for i in self.p_indice:
            for gt_f, img_f in zip(self.gt_files, self.img_files):
                pf_id = gt_f.split('_')[0]
                pf_id = pf_id[-3:]
                if i == int(pf_id):
                    img_volume = nib.load(self.d_p + img_f).get_fdata()
                    gt_volume = nib.load(self.gt_p + gt_f).get_fdata()
                    assert img_volume.shape == gt_volume.shape
                    depth = img_volume.shape[2]
                    for si in range(depth):
                        img = resize(img_volume[:, :, si], self.target_res_2d, anti_aliasing=True, preserve_range=True)
                        img = (img - img.min()) / (img.max() - img.min() + EPS)

                        gt = gt_volume[:, :, si]

                        gt_1_index = gt == 1
                        gt_2_index = gt == 2
                        gt_3_index = gt == 3
                        gt_4_index = gt == 4

                        gt_1 = gt * gt_1_index
                        gt_2 = gt * gt_2_index
                        gt_3 = gt * gt_3_index
                        gt_4 = gt * gt_4_index

                        gt_1 = resize(gt_1, self.target_res_2d, anti_aliasing=True, preserve_range=True)
                        gt_2 = resize(gt_2, self.target_res_2d, anti_aliasing=True, preserve_range=True)
                        gt_3 = resize(gt_3, self.target_res_2d, anti_aliasing=True, preserve_range=True)
                        gt_4 = resize(gt_4, self.target_res_2d, anti_aliasing=True, preserve_range=True)

                        self.img_samples.append(img[np.newaxis, :, :])
                        self.gt_samples.append(np.array([gt_1, gt_2, gt_3, gt_4]))
                        self.p_id.append(i)

    def __getitem__(self, item):

        return self.img_samples[item], self.gt_samples[item], self.p_id[item]

    def __len__(self):

        assert len(self.img_samples) == len(self.gt_samples)
        return len(self.img_samples)


class acdc_gan(Dataset):
    def __init__(self,
                 train_path='/home/data/jzheng/acdc/images/',
                 target_res=(32, 256, 256),
                 is_3d=False,
                 transforms=None
                 ):
        self.t_p = train_path
        self.files = os.listdir(self.t_p)
        self.sample_list_2d = []
        self.is_3d = is_3d
        self.target_res = target_res
        self.res_2d = (target_res[1], target_res[2])
        self.transforms = transforms

        if self.is_3d == False:
            for f in self.files:
                img = nib.load(self.t_p + f).get_fdata()
                depth = img.shape[2]
                f_i = int(round(depth * 0.1))
                b_i = int(round(depth * 0.9))
                interval_slice = img[:, :, f_i:b_i]
                for ii in range(interval_slice.shape[2]):
                    single_slice = interval_slice[:, :, ii]
                    single_slice = resize(single_slice, self.res_2d, anti_aliasing=True, preserve_range=True)
                    single_slice = (single_slice - single_slice.min()) / (single_slice.max() - single_slice.min() + EPS)
                    self.sample_list_2d.append(single_slice[None, :, :])

    def __len__(self):
        if self.is_3d == False:
            return len(self.sample_list_2d)
        else:
            return len(self.files)

    def __getitem__(self, index):
        if self.is_3d == False:
            return self.sample_list_2d[index], self.sample_list_2d[index]
        for f in self.files:
            img = nib.load(self.t_p + f).get_fdata()
            target_d_ratio = self.target_res[0] / img.shape[2]
            target_w_ratio = self.target_res[1] / img.shape[0]
            target_h_ratio = self.target_res[2] / img.shape[1]

            resize_img = zoom(img, (target_w_ratio, target_h_ratio, target_d_ratio))

            resize_img = np.swapaxes(resize_img, 0, 2)
            resize_img = np.swapaxes(resize_img, 1, 2)
            resize_img = (resize_img - resize_img.min()) / (resize_img.max() - resize_img.min() + EPS)
            if transforms != None:
                resize_img = self.transforms(resize_img)
            return resize_img, resize_img


class acdc_gan_single_slice(Dataset):
    def __init__(self, train_path='/well/papiez/shared/ACDC/clean_training/images/'):
        self.t_p = train_path
        self.files = os.listdir(self.t_p)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = self.files[index]
        img = nib.load(self.t_p + img).get_fdata()
        depth = img.shape[2]
        mid_d = int(depth / 2)
        mid_slice = img[:, :, mid_d]
        mid_slice = resize(mid_slice, (128, 128), anti_aliasing=True, preserve_range=True)
        mid_slice = (mid_slice - mid_slice.min()) / (mid_slice.max() - mid_slice.min() + EPS)
        # print(mid_slice.max(),mid_slice.min())

        return mid_slice, mid_slice


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, is_2d=False):
        self.is_2d = is_2d

    def __call__(self, sample):
        # image, label: For AHNet 2D to 3D,
        # 3D: WxHxD -> 1xWxHxD, 96x96x96 -> 1x96x96x96
        # 2D: WxHxD -> CxWxh, 224x224x3 -> 3x224x224
        image, label = sample['image'], sample['label']

        if self.is_2d:
            image = image.transpose(2, 0, 1).astype(np.float32)
            label = label.transpose(2, 0, 1)[1, :, :]
        else:
            # image = image.transpose(1, 0, 2)
            # label = label.transpose(1, 0, 2)
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)

        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}
