"""
ScanObjectNN download: http://103.24.77.34/scanobjectnn/h5_files.zip
"""

import os
import sys
import glob
import h5py,pickle
import numpy as np
from torch.utils.data import Dataset
import random, logging
import torch
import collections
from scipy.linalg import expm,norm
from pointnet2_ops import pointnet2_utils


def fps(data, number):
    '''
        data B N C
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number).long()
    fps_data = torch.gather(
        data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    return fps_data


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class PointCloudToTensor(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):

        data = torch.from_numpy(data).float()
        return data

class PointCloudCenterAndNormalize(object):
    def __init__(self, centering=True,
                 normalize=True,
                 **kwargs):
        self.centering = centering
        self.normalize = normalize

    def __call__(self, data):

        if self.centering:
            data = data - torch.mean(data, dim=0, keepdim=True)
        if self.normalize:
            m = torch.max(torch.sqrt(torch.sum(data ** 2, dim=-1, keepdim=True)), dim=0, keepdim=True)[0]
            data = data / m
        return data

class PointCloudScaling(object):
    def __init__(self,
                 scale=[2. / 3, 3. / 2],
                 anisotropic=True,
                 scale_xyz=[True, True, True],
                 mirror=[0, 0, 0],  # the possibility of mirroring. set to a negative value to not mirror
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.anisotropic = anisotropic
        self.scale_xyz = scale_xyz
        self.mirror = torch.from_numpy(np.array(mirror))
        self.use_mirroring = torch.sum(torch.tensor(self.mirror)>0) != 0

    def __call__(self, data):
        device = data.device
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32, device=device) * (
                self.scale_max - self.scale_min) + self.scale_min
        if self.use_mirroring:
            assert self.anisotropic==True
            self.mirror = self.mirror.to(device)
            mirror = (torch.rand(3, device=device) > self.mirror).to(torch.float32) * 2 - 1
            scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        data *= scale
        return data

class PointCloudRotation(object):
    def __init__(self, angle=[0, 0, 0], **kwargs):
        self.angle = np.array(angle) * np.pi

    @staticmethod
    def M(axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, data):

        device = data.device

        if isinstance(self.angle, collections.Iterable):
            rot_mats = []
            for axis_ind, rot_bound in enumerate(self.angle):
                theta = 0
                axis = np.zeros(3)
                axis[axis_ind] = 1
                if rot_bound is not None:
                    theta = np.random.uniform(-rot_bound, rot_bound)
                rot_mats.append(self.M(axis, theta))
            # Use random order
            np.random.shuffle(rot_mats)
            rot_mat = torch.tensor(rot_mats[0] @ rot_mats[1] @ rot_mats[2], dtype=torch.float32, device=device)
        else:
            raise ValueError()

        """ DEBUG
        from openpoints.dataset import vis_multi_points
        old_points = data.cpu().numpy()
        # old_points = data['pos'].numpy()
        # new_points = (data['pos'] @ rot_mat.T).numpy()
        new_points = (data @ rot_mat.T).cpu().numpy()
        vis_multi_points([old_points, new_points])
        End of DEBUG"""
        data = data @ rot_mat.T
        return data


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'h5_files')):
        # note that this link only contains the hardest perturbed variant (PB_T50_RS).
        # for full versions, consider the following link.
        www = 'https://web.northeastern.edu/smilelab/xuma/datasets/h5_files.zip'
        # www = 'http://103.24.77.34/scanobjectnn/h5_files.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_scanobjectnn_data(partition):
    #download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []

    h5_name = BASE_DIR + '/data/h5_files/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'
    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ScanObjectNN(Dataset):
    def __init__(self, num_points, partition='training'):
        data_dir = os.path.dirname(os.path.abspath(__file__))+ '/data/h5_files/main_split/'
        self.data, self.label = load_scanobjectnn_data(partition)
        if partition == 'test':
            precomputed_path = os.path.join(
                data_dir, f'{partition}_objectdataset_augmentedrot_scale75_1024_fps.pkl')
            if not os.path.exists(precomputed_path):
                data = torch.from_numpy(self.data).to(torch.float32).cuda()
                self.data = fps(data, 1024).cpu().numpy()
                with open(precomputed_path, 'wb') as f:
                    pickle.dump(self.data, f)
                    print(f"{precomputed_path} saved successfully")
            else:
                with open(precomputed_path, 'rb') as f:
                    self.data = pickle.load(f)
                    print(f"{precomputed_path} load successfully")
        logging.info(f'Successfully load ScanObjectNN {partition} '
                     f'size: {self.data.shape}, num_classes: {self.label.max() + 1}')
        self.num_points = num_points
        self.partition = partition
        self.pointcloudtotensor=PointCloudToTensor()
        self.pointcloudrotation = PointCloudRotation(angle=[0, 1, 0])
        self.pointcloudcenterandnormalize = PointCloudCenterAndNormalize()
        self.pointcloudscaling = PointCloudScaling(scale=[0.9, 1.1])
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'training':
            np.random.shuffle(pointcloud)
            pointcloud = self.pointcloudtotensor(pointcloud)
            pointcloud = self.pointcloudscaling(pointcloud)
            pointcloud = self.pointcloudcenterandnormalize(pointcloud)
            pointcloud = self.pointcloudrotation(pointcloud)
            pointcloud = torch.cat((pointcloud,pointcloud[:, 1:2] - pointcloud[:,1:2].min()),dim=1)
        if self.partition == 'test':
            pointcloud = self.pointcloudtotensor(pointcloud)
            pointcloud = self.pointcloudcenterandnormalize(pointcloud)
            pointcloud = torch.cat((pointcloud, pointcloud[:, 1:2] - pointcloud[:, 1:2].min()), dim=1)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ScanObjectNN(1024)
    test = ScanObjectNN(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label)