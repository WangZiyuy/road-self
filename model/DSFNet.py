from .basic_block import *
import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.backends.cudnn.benchmark = False

upsample = lambda x, scale: \
    F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=True)

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

class Unet_multistage(nn.Module):
    """
    模型共有三个encoder，分别输入遥感图像、轨迹特征、遥感图像
    """

    def __init__(self, conv1d=False):
        super(Unet_multistage, self).__init__()

        if conv1d:
            self.upsample = DecoderBlock1DConv4
        else:
            self.upsample = self.upsample_original

        self.conv_2_side = ConvReLU(128, 128, 3, 2, bn=True)
        self.conv_3_side = ConvReLU(256, 128, 3, 1, bn=True)
        self.conv_4_side = ConvReLU(512, 128, 3, 1, bn=True)
        self.conv_5_side = ConvReLU(1024, 128, 3, 1, bn=True)
        self.conv_fuse = ConvReLU(512, 128, 3, 1, bn=True)

        # traj特征编解码
        self.down1_traj = self.conv_stage(1, 64)
        self.down2_traj = self.conv_stage(64, 128)
        self.down3_traj = self.conv_stage(128, 256)
        self.down4_traj = self.conv_stage(256, 512)

        self.center_traj = self.conv_stage(512, 1024)

        self.up4_traj = self.conv_stage(1024, 512)
        self.up3_traj = self.conv_stage(512, 256)
        self.up2_traj = self.conv_stage(256, 128)
        self.up1_traj = self.conv_stage(128, 64)

        if conv1d:
            self.upsample = DecoderBlock1DConv4
        else:
            self.upsample = self.upsample_original

        self.trans4_traj = self.upsample(1024, 512)
        self.trans3_traj = self.upsample(512, 256)
        self.trans2_traj = self.upsample(256, 128)
        self.trans1_traj = self.upsample(128, 64)

        # 遥感轨迹特征编解码
        self.down1_src_traj = self.conv_stage(3, 64)
        self.down2_src_traj = self.conv_stage(64, 128)
        self.down3_src_traj = self.conv_stage(128, 256)
        self.down4_src_traj = self.conv_stage(256, 512)

        self.center_src_traj = self.conv_stage(512, 1024)

        self.up4_src_traj = self.conv_stage(1024, 512)
        self.up3_src_traj = self.conv_stage(512, 256)
        self.up2_src_traj = self.conv_stage(256, 128)
        self.up1_src_traj = self.conv_stage(128, 64)

        if conv1d:
            self.upsample = DecoderBlock1DConv4
        else:
            self.upsample = self.upsample_original

        self.trans4_src_traj = self.upsample(1024, 512)
        self.trans3_src_traj = self.upsample(512, 256)
        self.trans2_src_traj = self.upsample(256, 128)
        self.trans1_src_traj = self.upsample(128, 64)

        # traj主任务头
        self.traj_seg = nn.Sequential(
            ConvReLU(64, 64, 3, 2, bn=True),
            ConvReLU(64, 64, 3, 2, bn=True)
        )
        self.traj_conv_last = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        # self.road_seg = self.conv_stage(64, 64)
        self.road_seg = nn.Sequential(
            ConvReLU(64, 64, 3, 2, bn=True),
            ConvReLU(64, 64, 3, 2, bn=True)
        )
        self.conv_road_final = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.junc_seg = nn.Sequential(
            ConvReLU(64, 64, 3, 2, bn=True),
            ConvReLU(64, 64, 3, 2, bn=True)
        )
        self.conv_junc_final = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2)

        # self.init_weights()

        self.sfw1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.sfw2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.sfw3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.sfw4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.sfw1.data.fill_(0.25)
        self.sfw2.data.fill_(0.25)
        self.sfw3.data.fill_(0.25)
        self.sfw4.data.fill_(0.25)

        # 定义STFuse中co attention的融合特征图缩减通道
        self.ca_info_down1 = self.ca_conv_stage(2048, 1024)
        self.ca_info_down2 = self.ca_conv_stage(1024, 512)
        self.ca_info_down3 = self.ca_conv_stage(512, 256)
        self.ca_info_down4 = self.ca_conv_stage(256, 128)
        self.ca_info_down5 = self.ca_conv_stage(128, 64)

        self.ca_soft_down1 = self.ca_conv_stage(256, 2)
        self.ca_soft_down2 = self.ca_conv_stage(1024, 2)
        self.ca_soft_down3 = self.ca_conv_stage(4096, 2)
        self.ca_soft_down4 = self.ca_conv_stage(16384, 2)

        # 用于计算BTFuse输出
        self.temp = torch.ones((256, 256), device=torch.device("cuda"))

        # 用于STFuse关联矩阵增加权重
        self.W_b = nn.Parameter(torch.ones(1024, 1024))
        self.W_s = nn.Parameter(torch.ones(256, 1024))
        self.W_t = nn.Parameter(torch.ones(256, 1024))

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

    def gate_conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(),
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
        )

    def ca_map_upsampling(self):
        return nn.Sequential(
            nn.Upsample(scale_factor=2)
        )

    def ca_conv_stage(self, dim_in, dim_out, kernel_size=1, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
        )

    def upsample_original(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU()
        )

    # 加wb，计算FT与FI的关系矩阵，加上coattention公式45
    def co_att_first(self, feature1, feature2):
        fs1, ft1 = feature1, feature2
        B, N, W, H = fs1.shape
        info = torch.cat((fs1, ft1), 1)  # 除了权重还应生成一个特征图 [1, 2048, 16, 16]
        info = self.ca_info_down1(info)

        x1 = fs1.reshape(B, N, W * H)  # [1, 1024, 256]
        x2 = ft1.reshape(B, N, W * H)
        C = torch.bmm(x2.permute(0, 2, 1), torch.matmul(self.W_b, x1))  # （256，1024) * ((1024,1024)*(1024,256))
        Hs = nn.Tanh()(torch.matmul(self.W_s, x1) + torch.matmul(torch.matmul(self.W_t, x2), C))

        WI = F.softmax(self.ca_soft_down1(Hs.unsqueeze(-1)), dim=1).squeeze(-1)  # [1, 2, 256]
        WI1, WI2 = WI[:, 0, :].unsqueeze(1), WI[:, 1, :].unsqueeze(1)
        fs1_prime = (x1 * WI1 + x2 * WI2).reshape(B, N, W, H)  # [1, 1024, 256] [1, 1024, 16,16]

        return info, fs1_prime

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #             if m.bias is not None:
    #                 # m.weight.data.normal_(mean=0.0, std=1.0)
    #                 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    #                 m.bias.data.zero_()

    def forward(self, traj, src):
        """
        src:遥感图像
        traj:轨迹特征
        """
        tr_conv1_out = self.down1_traj(traj)
        tr_conv2_out = self.down2_traj(self.max_pool(tr_conv1_out)) # 256,256,64
        tr_conv3_out = self.down3_traj(self.max_pool(tr_conv2_out))
        tr_conv4_out = self.down4_traj(self.max_pool(tr_conv3_out))
        tr_center_out = self.center_traj(self.max_pool(tr_conv4_out))

        tr_out4 = self.up4_traj(torch.cat((self.trans4_traj(tr_center_out), tr_conv4_out), 1))
        tr_out3 = self.up3_traj(torch.cat((self.trans3_traj(tr_out4), tr_conv3_out), 1))
        tr_out2 = self.up2_traj(torch.cat((self.trans2_traj(tr_out3), tr_conv2_out), 1))
        tr_out1 = self.up1_traj(torch.cat((self.trans1_traj(tr_out2), tr_conv1_out), 1))
        tr_fts = self.traj_seg(tr_out1)
        tr_final = self.traj_conv_last(tr_fts)

        # 遥感图像道路提取编码器(SR)
        sr_conv1_out = self.down1_src_traj(src)
        sr_conv2_out = self.down2_src_traj(self.max_pool(sr_conv1_out))
        sr_conv2_out_side = self.conv_2_side(sr_conv2_out)
        sr_conv3_out = self.down3_src_traj(self.max_pool(sr_conv2_out))
        sr_conv3_out_side = self.conv_3_side(sr_conv3_out)
        sr_conv4_out = self.down4_src_traj(self.max_pool(sr_conv3_out))
        sr_conv4_out_side = self.conv_4_side(sr_conv4_out)
        sr_conv4_out_side = upsample(sr_conv4_out_side, 2)
        sr_center_out = self.center_src_traj(self.max_pool(sr_conv4_out))

        # 使用co attention的两个解码器交互
        fi1, ft1 = sr_center_out, tr_center_out  # [4, 1024, 16, 16]
        info1, fi_ca1 = self.co_att_first(fi1, ft1)

        sr_conv5_out_side = self.conv_5_side(fi_ca1)
        sr_conv5_out_side = upsample(sr_conv5_out_side, 4)

        stage_fuse = [sr_conv2_out_side, sr_conv3_out_side, sr_conv4_out_side, sr_conv5_out_side]
        stage_fuse = torch.cat(stage_fuse, dim=1)
        stage_fuse = self.conv_fuse(stage_fuse)
        # stage_fuse = sr_conv5_out_side

        sr_out4 = self.up4_src_traj(torch.cat((self.trans4_src_traj(fi_ca1), sr_conv4_out), 1))
        sr_out3 = self.up3_src_traj(torch.cat((self.trans3_src_traj(sr_out4), sr_conv3_out), 1))
        sr_out2 = self.up2_src_traj(torch.cat((self.trans2_src_traj(sr_out3), sr_conv2_out), 1))
        sr_out1 = self.up1_src_traj(torch.cat((self.trans1_src_traj(sr_out2), sr_conv1_out), 1))
        # torch.Size([1, 64, 256, 256])
        # 输出道路分割和路口分割结果
        road_fts = self.road_seg(sr_out1)
        road_final = self.conv_road_final(road_fts)

        junc_fts = self.junc_seg(sr_out1)
        junc_final = self.conv_junc_final(junc_fts)

        return sr_conv1_out, sr_conv2_out, sr_conv3_out, sr_conv4_out, stage_fuse, road_fts, junc_fts, road_final, junc_final, tr_fts, tr_final, fi_ca1


