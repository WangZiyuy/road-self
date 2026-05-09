import numpy as np
import torch
import torch.nn.init
from torch import nn
import torch.nn.functional as F
from .res2net import res2net50_v1b_26w_4s
from .DSFNet import Unet_multistage
from configs.config import config

upsample = lambda x, scale: \
    F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=True)

def print_memory_usage(stage_name):
    print(f"[{stage_name}] Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB")

class ConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sz, stride=1, relu=True, pd=True, bn=False):
        super(ConvReLU, self).__init__()
        padding = int((kernel_sz - 1) / 2) if pd else 0  # same spatial size by default
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_sz, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch) if bn else None  # eps=0.001, momentum=0, affine=True
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, ft_in, ft_out, out_ch):
        super(DecoderBlock, self).__init__()

        self.ft_conv = nn.Sequential(
            ConvReLU(ft_in, ft_out, 1))
        self.conv = nn.Sequential(
            ConvReLU(out_ch + ft_out, (out_ch + ft_out) // 2, 3),
            nn.Conv2d((out_ch + ft_out) // 2, out_ch, 3, padding=1))

    def forward(self, ft_cur, ft_pre):
        ft_cur = self.ft_conv(ft_cur)
        x = torch.cat((ft_cur, ft_pre), dim=1)
        x = self.conv(x)
        return x


class Hourglass(nn.Module):

    def __init__(self, input_ch, output_ch, ch=[32, 32, 32, 32]):
        super(Hourglass, self).__init__()
        ch = [input_ch] + ch
        self.encoder_1 = nn.Sequential(
            ConvReLU(ch[0], (ch[0]+ch[1])//2, 3),
            ConvReLU((ch[0]+ch[1])//2, ch[1], 3)
        )
        self.encoder_2 = nn.Sequential(
            ConvReLU(ch[1], (ch[1]+ch[2])//2, 3),
            ConvReLU((ch[1]+ch[2])//2, ch[2], 3)
        )
        self.encoder_3 = nn.Sequential(
            ConvReLU(ch[2], (ch[2]+ch[3])//2, 3),
            ConvReLU((ch[2]+ch[3])//2, ch[3], 3)
        )
        self.encoder_4 = nn.Sequential(
            ConvReLU(ch[3], (ch[3]+ch[4])//2, 3),
            ConvReLU((ch[3]+ch[4])//2, ch[4], 3)
        )
        self.encoder_5 = nn.Sequential(
            ConvReLU(ch[4], ch[4], 3),
            ConvReLU(ch[4], ch[4], 3)
        )
        self.decoder_4 = nn.Sequential(
            ConvReLU(ch[4], (ch[3]+ch[4])//2, 3),
            ConvReLU((ch[3]+ch[4])//2, ch[3], 3)
        )
        self.decoder_3 = nn.Sequential(
            ConvReLU(ch[3], (ch[2]+ch[3])//2, 3),
            ConvReLU((ch[2]+ch[3])//2, ch[2], 3)
        )
        self.decoder_2 = nn.Sequential(
            ConvReLU(ch[2], (ch[1]+ch[2])//2, 3),
            ConvReLU((ch[1]+ch[2])//2, ch[1], 3)
        )
        self.decoder_1 = nn.Sequential(
            ConvReLU(ch[1], (ch[0]+ch[1])//2, 3),
            ConvReLU((ch[0]+ch[1])//2, output_ch, 3)
        )
        self.maxpool = nn.MaxPool2d(2, 2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoder_1 = self.encoder_1(x)
        encoder_1_pool = self.maxpool(encoder_1)
        encoder_2 = self.encoder_2(encoder_1_pool)
        encoder_2_pool = self.maxpool(encoder_2)
        encoder_3 = self.encoder_3(encoder_2_pool)
        encoder_3_pool = self.maxpool(encoder_3)
        encoder_4 = self.encoder_4(encoder_3_pool)
        encoder_4_pool = self.maxpool(encoder_4)
        encoder_5 = self.encoder_5(encoder_4_pool)
        decoder_5_up = upsample(encoder_5, 2) + encoder_4
        decoder_4 = self.decoder_4(decoder_5_up)
        decoder_4_up = upsample(decoder_4, 2) + encoder_3
        decoder_3 = self.decoder_3(decoder_4_up)
        decoder_3_up = upsample(decoder_3, 2) + encoder_2
        decoder_2 = self.decoder_2(decoder_3_up)
        decoder_2_up = upsample(decoder_2, 2) + encoder_1
        decoder_1 = self.decoder_1(decoder_2_up)
        return decoder_1


class DirectionalFilteringModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DirectionalFilteringModule, self).__init__()
        # 卷积层用于特征提取
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        # 用于方向预测的卷积层
        self.direction_conv = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        # 特征提取
        feature_map = F.relu(self.conv(x))

        # 方向预测，输出方向图
        direction_map = self.direction_conv(x)
        direction_map = F.softmax(direction_map, dim=1)  # 假设方向预测为分类问题

        # 生成注意力图，基于方向图的预测
        attention_map = self.create_attention_map(direction_map)

        # 将注意力图应用于特征图，进行过滤
        filtered_features = feature_map * attention_map

        return filtered_features

    def create_attention_map(self, direction_map):
        # 通过方向图生成一个注意力图，基于预测的方向信息
        # 假设方向图的值为0-1之间的概率，可以根据方向图的输出生成过滤的注意力图
        attention_map = torch.sigmoid(direction_map)  # 可以根据需要调整这里的激活函数
        return attention_map


class Transformer(nn.Module):
    def  __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim):
        """
        :param input_dim: 输入特征的维度
        :param embed_dim: 嵌入维度
        :param num_heads: Transformer 中的头数
        :param num_layers: Transformer 编码器层数
        :param output_dim: 输出特征的维度

        Transformer 接受的输入格式通常是一个 (batch_size, seq_len, feature_dim) 的张量，其中 seq_len 是序列长度，feature_dim 是每个时间步（或轨迹点）的特征维度
        """
        super(Transformer, self).__init__()
        # 输入的线性映射
        self.embedding = nn.Linear(input_dim, embed_dim)
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer1 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 最后的输出层
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x, src_key_padding_mask):
        """
        :param x: 输入的特征，形状为 (batch_size, seq_len, input_dim)
        :return: 输出特征，形状为 (batch_size, output_dim)
        """
        # 对输入进行嵌入
        x = self.embedding(x)
        # Transformer 编码器处理
        x = self.transformer1(x, src_key_padding_mask=src_key_padding_mask)
        # 取 Transformer 输出的最后一个时间步的特征
        x = x[:, -1, :]
        # 输出层
        x = self.fc(x)
        return x


class RPNet(nn.Module):

    def __init__(self, num_targets=4):
        super(RPNet, self).__init__()

        self.num_targets = num_targets

        self.conv_2_side = ConvReLU(256, 128, 3, 1, bn=True)
        self.conv_3_side = ConvReLU(512, 128, 3, 1, bn=True)
        self.conv_4_side = ConvReLU(1024, 128, 3, 1, bn=True)
        self.conv_5_side = ConvReLU(2048, 128, 3, 1, bn=True)
        self.conv_fuse = ConvReLU(512, 128, 3, 1, bn=True)
        self.avgpool4 = nn.AvgPool2d(4, 4).cuda()

        # self.directional_filtering = DirectionalFilteringModule(512, 256)

        self.transformer = Transformer(2, 32, num_heads=4, num_layers=1, output_dim=64)
        self.road_seg = nn.Sequential(
            ConvReLU(128, 64, 3, 1, bn=True),
            ConvReLU(64, 64, 1, 1, bn=True)
        )
        self.conv_road_final = nn.Conv2d(64, 1, 1, 1, 0)

        self.junc_seg = nn.Sequential(
            ConvReLU(128, 64, 3, 1, bn=True),
            ConvReLU(64, 64, 1, 1, bn=True)
        )
        self.conv_junc_final = nn.Conv2d(64, 1, 1, 1, 0)

        self.fuse_module = Hourglass(
            128  + 64 + 64 + 32 * (self.num_targets-1) + 1,  # 353
            32, [128, 128, 128, 128])
        self.fuse_module_traj = Hourglass(
            128  + 64 + 64 + 64 + 32 * (self.num_targets-1) + 1,  # 353
            32, [128, 128, 128, 128])


        self.ft_chs = [1024, 512, 256, 64]
        self.decoders = nn.ModuleList([
            DecoderBlock(self.ft_chs[0], 32, 32),
            DecoderBlock(self.ft_chs[1], 32, 32),
            DecoderBlock(self.ft_chs[2], 32, 32),
            DecoderBlock(self.ft_chs[3], 32, 32),
        ])

        self.ft_chs_DFS = [512, 256, 128, 64]
        self.decoders_DFS = nn.ModuleList([
            DecoderBlock(self.ft_chs_DFS[0], 32, 32),
            DecoderBlock(self.ft_chs_DFS[1], 32, 32),
            DecoderBlock(self.ft_chs_DFS[2], 32, 32),
            DecoderBlock(self.ft_chs_DFS[3], 32, 32),
        ])
        self.next_step_final = nn.Conv2d(32, 1, 1, 1, 0)
        self.conv_final = nn.Conv2d(32, 1, 3, 1, 1)

        # self.feature_projection = nn.Conv2d(1, 64, 1, 1)
        self.upsample1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=64, stride=64)
        self.upsample2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=4)
        self.next_step_256fuse = nn.Conv2d(32, 32, 1, 1, 0)


        self.init_weights()
        ## first init_weights for added parts, then init res2net
        res2net = res2net50_v1b_26w_4s(pretrained=True)

        self.stage_1 = nn.Sequential(
            res2net.conv1,
            res2net.relu)
        self.stage_1_traj = nn.Sequential(
        # 修改第一层卷积层，使其能够处理单通道输入
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            res2net.relu)
        self.stage_1_traj_aerial = nn.Sequential(
            # 修改第一层卷积层，使其能够处理单通道输入
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
            res2net.relu)
        self.maxpool = res2net.maxpool
        self.stage_2 = res2net.layer1
        self.stage_3 = res2net.layer2
        self.stage_4 = res2net.layer3
        self.stage_5 = res2net.layer4

        self.DSF = Unet_multistage()

        self.up4_anchor = self.conv_stage(1024, 512)
        self.up3_anchor = self.conv_stage(512, 256)
        self.up2_anchor = self.conv_stage(256, 128)
        self.up1_anchor = self.conv_stage(128, 64)
        self.up0_anchor = self.conv_stage(64, 32)

        self.trans4_anchor = self.DSFupsample(1024, 512)
        self.trans3_anchor = self.DSFupsample(512, 256)
        self.trans2_anchor = self.DSFupsample(256, 128)
        self.trans1_anchor = self.DSFupsample(128, 64)

        self.missing_traj_feature = nn.Parameter(torch.zeros(1, 64, 256, 256))
        nn.init.normal_(self.missing_traj_feature, mean=0.0, std=0.02)

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),

            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        )

    def DSFupsample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, aerial_image, traj_image, aerial_traj_image, neighborhood_trajectory_norm, valid_mask, walked_path, NUM_TARGETS=None, test=False, model=None, use_traj=None):

        # 首先检查neighborhood_trajectory是否为空
        # if neighborhood_trajectory_norm is None:
        #     have_neighborhood_trajectory = False
        # else:
        #     if torch.all(neighborhood_trajectory_norm == 0):
        #         have_neighborhood_trajectory = False  # 标记是否使用轨迹
        #     else:
        #         have_neighborhood_trajectory = True

        if neighborhood_trajectory_norm is None or torch.all(neighborhood_trajectory_norm == 0):
            have_neighborhood_trajectory = False
        else:
            have_neighborhood_trajectory = True

        road_final, junc_final, traj_final = None, None, None
        road_fts, junc_fts, traj_fts = None, None, None
        stage_fuse = None
        
        # 用于存储中间特征图的字典
        feature_maps = {}

        if model == 'origin':
            stage_1 = self.stage_1(aerial_image)
            stage_1_down = self.maxpool(stage_1)

            stage_2 = self.stage_2(stage_1_down)
            stage_2_side = self.conv_2_side(stage_2)
            stage_2_side = upsample(stage_2_side, 4)

            stage_3 = self.stage_3(stage_2)
            stage_3_side = self.conv_3_side(stage_3)
            stage_3_side = upsample(stage_3_side, 8)

            stage_4 = self.stage_4(stage_3)
            stage_4_side = self.conv_4_side(stage_4)
            stage_4_side = upsample(stage_4_side, 8)

            stage_5 = self.stage_5(stage_4)
            stage_5_side = self.conv_5_side(stage_5)
            stage_5_side = upsample(stage_5_side, 8)
            stage_fuse = [stage_2_side, stage_3_side, stage_4_side, stage_5_side]
            stage_fuse = torch.cat(stage_fuse, dim=1)
            # torch.Size([5, 512, 128, 128])

            stage_fuse = self.conv_fuse(stage_fuse)
            # torch.Size([5, 128, 128, 128])

            # 输出道路分割和路口分割结果
            road_fts = self.road_seg(stage_fuse)
            road_final = self.conv_road_final(road_fts)

            junc_fts = self.junc_seg(stage_fuse)
            junc_final = self.conv_junc_final(junc_fts)
            
            # 保存中间特征图
            feature_maps['stage_1'] = stage_1
            feature_maps['stage_2'] = stage_2
            feature_maps['stage_3'] = stage_3
            feature_maps['stage_4'] = stage_4
            feature_maps['stage_5'] = stage_5
            feature_maps['stage_fuse'] = stage_fuse
            feature_maps['road_fts'] = road_fts
            feature_maps['junc_fts'] = junc_fts

        elif model == 'DSFNet':
            #所以DSFNet的功能只能用来进行前置辅助信息的生成（道路和节点） 另一个生成anchor的解码器和DSFNet就无关了
            print('using DSFNet')
            stage_1_DFS, stage_2_DFS, stage_3_DFS, stage_4_DFS, stage_fuse, road_fts, junc_fts, road_final, junc_final, traj_fts, traj_final, fi_ca1 = self.DSF(traj_image, aerial_image)
            
            # 保存DSFNet的中间特征图
            feature_maps['stage_1_DFS'] = stage_1_DFS
            feature_maps['stage_2_DFS'] = stage_2_DFS
            feature_maps['stage_3_DFS'] = stage_3_DFS
            feature_maps['stage_4_DFS'] = stage_4_DFS
            feature_maps['stage_fuse'] = stage_fuse
            feature_maps['road_fts'] = road_fts
            feature_maps['junc_fts'] = junc_fts
            feature_maps['traj_fts'] = traj_fts
            feature_maps['fi_ca1'] = fi_ca1

# ______________________________________________________________________________________________________
# ______________________________________________________________________________________________________

        # -------------------- 轨迹特征（推理也计算并返回） --------------------
        # 在 test 模式下也需要可视化 valid_traj_feature，因此将其前置计算
        if have_neighborhood_trajectory and use_traj:
            # neighborhood_trajectory_norm: [B, K, L, 2], 展平成序列送入 Transformer
            neighborhood_trajectory = neighborhood_trajectory_norm.view(
                neighborhood_trajectory_norm.size(0), -1, neighborhood_trajectory_norm.size(3))
            # Transformer(batch_first=True): src_key_padding_mask 形状应为 (B, L)，True 表示需要被mask
            padding_mask = (~valid_mask).view(neighborhood_trajectory_norm.size(0), -1)
            valid_traj_feature = self.transformer(
                neighborhood_trajectory, src_key_padding_mask=padding_mask)
            valid_traj_feature = valid_traj_feature.unsqueeze(2).unsqueeze(3)
            valid_traj_feature = self.upsample1(valid_traj_feature)
            valid_traj_feature = self.upsample2(valid_traj_feature)
            valid_traj_feature = F.interpolate(
                valid_traj_feature, size=(128, 128), mode='bilinear', align_corners=False)
            # 轻度高斯平滑，减少噪声与边界伪影
            try:
                channels = valid_traj_feature.shape[1]
                gaussian_kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=valid_traj_feature.device)
                gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
                weight = gaussian_kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
                valid_traj_feature = F.conv2d(valid_traj_feature, weight, padding=1, groups=channels)
            except Exception:
                pass
        elif not have_neighborhood_trajectory and use_traj:
            B = aerial_image.size(0)
            valid_traj_feature = self.missing_traj_feature.expand(B, -1, -1, -1)
            valid_traj_feature = F.interpolate(
                valid_traj_feature, size=(128, 128), mode='bilinear', align_corners=False)
            try:
                channels = valid_traj_feature.shape[1]
                gaussian_kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=valid_traj_feature.device)
                gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
                weight = gaussian_kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
                valid_traj_feature = F.conv2d(valid_traj_feature, weight, padding=1, groups=channels)
            except Exception:
                pass

        # 推理也保证 feature_maps 中一定包含 valid_traj_feature 键
        if 'valid_traj_feature' in locals():
            feature_maps['valid_traj_feature'] = valid_traj_feature
        else:
            B = aerial_image.size(0)
            placeholder_traj_feature = torch.zeros(B, 64, 128, 128).cuda()
            feature_maps['valid_traj_feature'] = placeholder_traj_feature

        if test:
            # 生成“融合后特征”的可视化预览，便于对比有/无轨迹的影响
            try:
                B = stage_fuse.shape[0]
                H, W = stage_fuse.shape[2], stage_fuse.shape[3]
                walked_path_vis = walked_path if walked_path is not None else torch.zeros(B, 1, H, W, device=stage_fuse.device)
                next_points_placeholder_vis = torch.zeros(
                    B, 32 * (self.num_targets - 1), H, W, device=stage_fuse.device)

                if use_traj:
                    preview_in = torch.cat(
                        [stage_fuse, road_fts, junc_fts, walked_path_vis, feature_maps['valid_traj_feature'], next_points_placeholder_vis], dim=1)
                    stage_fuse_after = self.fuse_module_traj(preview_in)
                else:
                    preview_in = torch.cat(
                        [stage_fuse, road_fts, junc_fts, walked_path_vis, next_points_placeholder_vis], dim=1)
                    stage_fuse_after = self.fuse_module(preview_in)

                feature_maps['stage_fuse_after'] = stage_fuse_after
            except Exception:
                # 预览失败不影响主流程
                pass

            # 推理返回：道路/路口结果 + 中间特征（含 valid_traj_feature 与 stage_fuse_after）
            return {
                'road': road_final,
                'junc': junc_final,
                'feature_maps': feature_maps
            }

        # 用于存储锚点结果
        next_points_placeholder = torch.zeros(
            (stage_fuse.shape[0],
             32 * (self.num_targets-1),
             stage_fuse.shape[2],
             stage_fuse.shape[3])).cuda()

        # 训练/解码阶段后续使用的 valid_traj_feature 在上面已计算，此处不再重复

        # 在第二维度拼接融合信息、道路最终特征、路口最终特征、路径信息
        # print("stage_fuse shape: ", stage_fuse.shape, road_fts.shape, junc_fts.shape, walked_path.shape, next_points_placeholder.shape)
        # stage_fuse shape: torch.Size([5, 128, 64, 64]) torch.Size([5, 64, 64, 64]) torch.Size([5, 64, 64, 64]) torch.Size([5, 1, 64, 64]) torch.Size([5, 96, 64, 64])
        if have_neighborhood_trajectory and use_traj:
            print("Info: Valid trajectory data detected in the batch. Using trajectory-enhanced features.")
            stage_fuse = torch.cat([stage_fuse, road_fts, junc_fts, walked_path, valid_traj_feature, next_points_placeholder], dim=1)
        elif not have_neighborhood_trajectory and use_traj:
            # 如果没有轨迹数据，只使用遥感图像相关数据进行处理
            print("Warning: No valid trajectory data in the batch. Using trimmed features.")
            stage_fuse = torch.cat([stage_fuse, road_fts, junc_fts, walked_path, valid_traj_feature, next_points_placeholder], dim=1)
        else:
            stage_fuse = torch.cat([stage_fuse, road_fts, junc_fts, walked_path, next_points_placeholder], dim=1)

        if self.training:
            stage_fuse_list = [stage_fuse]

        anchor_fts = None
        next_points = []
        next_points_lowrs = []  # low resolution

        for i in range(NUM_TARGETS if
                       NUM_TARGETS is not None else self.num_targets):
            if self.training:
                if use_traj:
                    next_step = self.fuse_module_traj(stage_fuse_list[i])
                else:
                    next_step = self.fuse_module(stage_fuse_list[i])
            else:
                if use_traj:
                    next_step = self.fuse_module_traj(stage_fuse)
                else:
                    next_step = self.fuse_module(stage_fuse)
            # 为了适配256，256，先降维再1×1卷积
            next_step = upsample(next_step, 0.25)
            next_step = self.next_step_256fuse(next_step)
            next_points_lowrs.append(upsample(self.next_step_final(next_step),4))

            if model == 'origin':
                # stage1-4 (64,128,128) (256,64,64) (512,32,32) (1024,32,32)
                decoded_ft_4 = self.decoders[0](upsample(stage_4, 2), next_step)
                decoded_ft_3 = self.decoders[1](upsample(stage_3, 2), decoded_ft_4)
                decoded_ft_2 = self.decoders[2](upsample(stage_2, 2), upsample(decoded_ft_3, 2))
                decoded_ft_1 = self.decoders[3](upsample(stage_1, 2), upsample(decoded_ft_2, 2))
            elif model == 'DSFNet':
                # TODO 这里也改成Unet的解码器结构 应该更合理
                decoded_ft_4 = self.up4_anchor(torch.cat((self.trans4_anchor(fi_ca1), stage_4_DFS), 1))
                decoded_ft_3 = self.up3_anchor(torch.cat((self.trans3_anchor(decoded_ft_4), stage_3_DFS), 1))
                decoded_ft_2 = self.up2_anchor(torch.cat((self.trans2_anchor(decoded_ft_3), stage_2_DFS), 1))
                decoded_ft_1 = self.up1_anchor(torch.cat((self.trans1_anchor(decoded_ft_2), stage_1_DFS), 1))
                decoded_ft_1 = self.up0_anchor(decoded_ft_1)
            
            # 保存解码器特征图
            feature_maps[f'decoded_ft_4_step_{i}'] = decoded_ft_4
            feature_maps[f'decoded_ft_3_step_{i}'] = decoded_ft_3
            feature_maps[f'decoded_ft_2_step_{i}'] = decoded_ft_2
            feature_maps[f'decoded_ft_1_step_{i}'] = decoded_ft_1

            ch_idx = -(self.num_targets - i - 1) * 32

            if i < self.num_targets - 1:
                if anchor_fts is None:
                    # anchor_fts = self.avgpool4(decoded_ft_1)
                    anchor_fts = decoded_ft_1
                else:
                    # 叠加锚点结果
                    # anchor_fts += self.avgpool4(decoded_ft_1)
                    anchor_fts = anchor_fts + decoded_ft_1
                if self.training:
                    stage_fuse_list.append(stage_fuse_list[i].clone())
                    # 在第二个维度，用 32*锚点数 个元素来记录锚点值
                    stage_fuse_list[i+1][:, ch_idx:ch_idx+32 if ch_idx+32 != 0 else None, :, :] = anchor_fts
                else:
                    stage_fuse[:, ch_idx:ch_idx+32 if ch_idx+32 != 0 else None, :, :] = anchor_fts
                
                # 保存anchor_fts
                feature_maps[f'anchor_fts_step_{i}'] = anchor_fts

            decoded_ft_1 = self.conv_final(decoded_ft_1)
            next_points.append(decoded_ft_1)

        next_points = torch.cat(next_points, dim=1)  # torch.Size([4, 4, 256, 256])
        next_points_lowrs = torch.cat(next_points_lowrs, dim=1)
        return {
            'road': road_final,
            'junc': junc_final,
            'anchor': next_points,
            'anchor_lowrs': next_points_lowrs,
            'traj_road':traj_final,
            'feature_maps': feature_maps,  # 添加特征图字典到返回值
        }


def build_model(num_targets=4):
    return RPNet(num_targets=num_targets)


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = torch.nn.DataParallel(build_model()).cuda().eval()
    batch_size = 12
    input_img = torch.zeros((batch_size, 3, 256, 256)).cuda()
    input_walked_path = torch.zeros((batch_size, 1, 64, 64)).cuda()
    model(input_img, input_walked_path)
    print('Memory useage: %.4fM' % (torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.4fM' % (total / 1e6))
