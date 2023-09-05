import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_base.utils.builder import build
from vision_base.networks.blocks.blocks import Scale, GradientScale, ConvBnReLU
from vision_base.networks.blocks.blocks import LinearLNDropout
from panoptic_bev.model import NaiveBEVMetaArch

def get_coordinates(x_range, y_range, z_range):
    coordinates = torch.stack(torch.meshgrid(
        torch.arange(x_range[0], x_range[1], x_range[2]),
        - torch.arange(y_range[0], y_range[1], y_range[2]), # this minus is caused by the dataset setting
        torch.arange(z_range[0], z_range[1], z_range[2]),
    ), dim=-1).cuda()# [X, Y, Z, 3]
    return coordinates

class Neck(nn.Module):
    def __init__(self, num_input_features=[64, 128, 256], num_output_features=256):
        super(Neck, self).__init__()
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(num_input_features[i], num_output_features, 1)
                for i in range(len(num_input_features))
            ]
        )
        self.upsamples = nn.ModuleList(
            [nn.Upsample(scale_factor=2), nn.Upsample(scale_factor=4)]
        )
    
    def forward(self, feat_list):
        assert len(feat_list) == len(self.num_input_features)
        outputs = [conv(feat) for conv, feat in zip(self.convs, feat_list)]
        output = outputs[0] +\
                 self.upsamples[0](outputs[1]) +\
                 self.upsamples[1](outputs[2])
        return output


class GridSample(nn.Module):
    def __init__(self, 
                x = [0, 112.64, 0.16],
                y = [-61.44, 61.44, 0.16],
                z = [-2.5, 1.5, 0.5],
                offset_branch=3, embedding_size=64, sample_iteration=3):
        super(GridSample, self).__init__()
        self.offset_branch = offset_branch
        self.embedding_size = embedding_size
        self.sample_iteration = sample_iteration

        grid_coordinates = get_coordinates(x, y, z) # [X, Y, Z, 3]
        x_size, y_size, z_size, _ = grid_coordinates.shape
        self.register_buffer('homo_grid_coordinates', 
                             torch.cat([
                                     grid_coordinates,
                                     torch.ones_like(grid_coordinates[..., 0:1])
                                 ], dim=-1)[None, ]
                            ) #[1, X, Y, Z, 4]
        bev_embedding = torch.zeros([x_size, y_size, z_size, self.embedding_size])
        self.register_parameter('bev_embedding', nn.Parameter(bev_embedding))
        self.offset_projs = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(self.embedding_size, self.offset_branch * 2),
                Scale(0.1), GradientScale(0.1)
            ) for _ in range(self.sample_iteration)]
        )
        
        self.weight_projs = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(self.embedding_size, self.offset_branch),
                nn.Softmax(dim=-1)) for _ in range(self.sample_iteration)]
        )

        self.linear_projs = nn.ModuleList(
            [LinearLNDropout(self.embedding_size, self.embedding_size, 0.5)
            for _ in range(self.sample_iteration)]
        )
        

    def forward(self, data_dict, img_feat):
        T_velo2cam = data_dict['T_velo2cam']
        intrinsic = data_dict['intrinsic']
        T_velo2image = torch.matmul(intrinsic, T_velo2cam) #[B, 3, 4]
        image_coordinates = torch.matmul(
            T_velo2image.reshape([-1, 1, 1, 1, 3, 4]), # [B, 1, 1, 1, 3, 4]
            self.homo_grid_coordinates.unsqueeze(-1) # [1, X, Y, Z, 4, 1]
        ).squeeze(-1) # [B, X, Y, Z, 3]
        B, X, Y, Z, _ = image_coordinates.shape
        # image_coordinate_clean : [B, X, Y, Z, 2]
        image_coordinate_clean = image_coordinates[..., 0:2] / image_coordinates[..., 2:3]
        H, W = data_dict['image'].shape[2:]
        half_shapes = torch.tensor([W, H]).float().cuda() / 2
        image_coordinate_clean = image_coordinate_clean / half_shapes - 1

        image_coordinate_clean = image_coordinate_clean.reshape([B, -1, 1, 2]) # [B, XYZ, 1, 2]
        bev_embedding = self.bev_embedding.reshape([1, -1, 1, self.embedding_size]) # [1, XYZ, 1, C]
        XYZ = X * Y * Z

        for i in range(self.sample_iteration):
            offsets = self.offset_projs[i](bev_embedding).reshape(-1, XYZ, self.offset_branch, 2) # [1, XYZ, b, 2]
            weights = self.weight_projs[i](bev_embedding).squeeze(2).unsqueeze(1) # [1, 1, XYZ, b] / [B, 1, XYZ, b]
            feat = F.grid_sample(
                img_feat, 
                image_coordinate_clean + offsets, # [B, XYZ, b, 2]
            ) # [B, C, XYZ, b]
            feat = torch.sum(feat * weights, dim=-1) # [B, C, XYZ]
            bev_embedding = bev_embedding + feat.permute([0, 2, 1]).unsqueeze(2).contiguous() # [B, XYZ, 1, C]
            bev_embedding = bev_embedding + self.linear_projs[i](bev_embedding)

        bev_embedding = bev_embedding.reshape([B, X, Y, Z, self.embedding_size])
        # Average over Z axis, and permute to [BCYX] for dataset alignment
        bev_embedding = bev_embedding.permute([0, 4, 2, 1, 3]).contiguous().mean(-1) # [B, C, Y, X] 
        return bev_embedding

class GridSampleMetaArch(NaiveBEVMetaArch):
    def _build_model(self):
        self.backbone = build(**self.network_cfg.backbone_cfg)
        self.transform = GridSample(**self.network_cfg.transform_cfg)
        bev_embedding_size = self.transform.embedding_size
        output_channel = self.network_cfg.output_channel
        self.upsample = nn.Sequential(
            ConvBnReLU(bev_embedding_size, bev_embedding_size, (3, 3)),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnReLU(bev_embedding_size, bev_embedding_size, (3, 3)),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(bev_embedding_size, output_channel, 1),
            # nn.Upsample(scale_factor=4)
        )
        self.neck = Neck(num_input_features=[512, 1024, 2048], num_output_features=bev_embedding_size)

    def _inference(self, data, meta):
        img_batch = data['image']
        N, C, H, W = img_batch.shape
        feats = self.backbone(img_batch) #[12 * 4]
        feat = self.neck(feats)
        bev_features = self.transform(data, feat)
        output = self.upsample(bev_features)
        return output
