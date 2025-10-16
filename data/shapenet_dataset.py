import os
import random
import torch
import torchvision.transforms.functional as TF
import numpy as np
from scipy.io import loadmat
from PIL import Image
from data.base_dataset import BaseDataset
import soft_renderer as sr
import soft_renderer.functional as srf


class ShapeNetDataset(BaseDataset):
    """
    Dataset for loading ShapeNet-Synthetic data.
    ShapeNet-Sythetic is used for training and evaluation.
    """
    def modify_commandline_options(parser):
        parser.add_argument('--n_views_per_obj', type=int, default=20)
        return parser

    def __init__(self, opt, mode):
        super().__init__(opt, mode)
        self.opt = opt
        self.split = mode
        assert self.split in ['train', 'val', 'test']
        # 支持多个类别
        self.class_ids = opt.class_id if isinstance(opt.class_id, list) else [opt.class_id]
        self.dat = []

        for class_id in self.class_ids:
            class_root = os.path.join(opt.dataset_root, class_id)
            list_path = os.path.join(class_root, self.split + '.lst')
            if not os.path.exists(list_path):
                print(f"Warning: list file {list_path} not found, skipping class {class_id}")
                continue
            with open(list_path) as f:
                obj_ids = list(filter(None, f.read().split('\n')))

            for obj_id in obj_ids:
                obj_path = os.path.join(class_root, obj_id)
                obj_dat = []
                if os.path.exists(os.path.join(obj_path, 'view.txt')):
                    with open(os.path.join(obj_path, 'view.txt')) as f:
                        obj_cameras = [list(map(float, c.split(' '))) for c in list(filter(None, f.read().split('\n')))]
                else:
                    continue
                view_feat_path = os.path.join(obj_path, 'view_txt.pt')
                if os.path.exists(view_feat_path):
                    view_feats = torch.load(view_feat_path, weights_only=True)  # shape: [N, 512]
                    # view_feats[i] 是第 i 个视角的文本特征
                else:
                    continue

                for i in range(opt.n_views_per_obj):
                    obj_camera = obj_cameras[i]
                    view_feat =view_feats[i]
                    elevation, azimuth = self.get_view_tensor(obj_camera[1], obj_camera[0])
                    obj_dat.append({
                        'class_id': class_id,
                        'obj_id': obj_id,
                        'image': os.path.join(obj_path, 'sketches', f"render_{i}.png"),
                        'camera': self.get_camera_tensor(obj_camera[3], obj_camera[1], obj_camera[0]),
                        'elevation': elevation,
                        'azimuth': azimuth,
                        'voxel': os.path.join(obj_path, 'voxel.mat'),
                        'view_feat':view_feat,
                    })
                self.dat.append(obj_dat)
        
    def __len__(self):
        if self.split == 'train':
            return len(self.dat)
        else:
            return len(self.dat) * self.opt.n_views_per_obj
    
    def __getitem__(self, index):
        if self.split == 'train':
            obj_dat = self.dat[index]
            view_dat = random.sample(obj_dat, k=2)
            assert view_dat[0]['class_id'] == view_dat[1]['class_id'], "两个视角的类别不一致！"

            class_id = self.class_ids.index(view_dat[0]['class_id'])  # 转成 int label

            camera = (view_dat[0]['camera'], view_dat[1]['camera'])
            image = (self.get_image_tensor(view_dat[0]['image']), self.get_image_tensor(view_dat[1]['image']))
            elevation = (view_dat[0]['elevation'], view_dat[1]['elevation'])
            azimuth = (view_dat[0]['azimuth'], view_dat[1]['azimuth'])
            return {
                'image': image,
                'camera': camera,
                'elevation': elevation,
                'azimuth': azimuth,
                'class_id':torch.tensor(class_id).long(),
                'view_feat': (
                                view_dat[0]['view_feat'],
                                view_dat[1]['view_feat'])
            }
        else:
            obj_dat = self.dat[index // self.opt.n_views_per_obj]
            view_dat = obj_dat[index % self.opt.n_views_per_obj]

            class_id = self.class_ids.index(view_dat['class_id'])  # 转成 int label
            camera = view_dat['camera']
            image = self.get_image_tensor(view_dat['image'])
            elevation, azimuth = view_dat['elevation'], view_dat['azimuth']
            voxel = self.get_voxel_tensor(view_dat['voxel'])
            return {
                'image': image,
                'camera': camera,
                'elevation': elevation,
                'azimuth': azimuth,
                'voxel': voxel,
                'class_id': torch.tensor(class_id).long(),
                'view_feat':view_dat['view_feat']
            }

    def get_image_tensor(self, path):
        image = Image.open(path).convert('RGBA')
        image = TF.resize(image, (self.opt.image_size, self.opt.image_size))
        image = TF.to_tensor(image)
        torch.where(image[3] > 0.5, torch.tensor(1.), torch.tensor(0.))
        return image
    
    def get_camera_tensor(self, distance, elevation, azimuth):
        camera = srf.get_points_from_angles(distance, elevation, azimuth)
        return torch.Tensor(camera).float()

    def get_voxel_tensor(self, path):
        # ground truth voxel head to x, up to y
        # transform to be able to compare with the voxel converted by SoftRas
        voxel = loadmat(path)['Volume'].astype(np.float32)
        voxel = np.rot90(np.rot90(voxel, axes=(1, 0)), axes=(2, 1))
        voxel = torch.from_numpy(voxel)
        return voxel

    def get_view_tensor(self, elevation, azimuth):
        return torch.tensor(elevation, dtype=torch.float32), torch.tensor(azimuth, dtype=torch.float32)
