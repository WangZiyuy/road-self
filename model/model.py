import torch
import torch.nn.init
from torch import nn
import torch.nn.functional as F

from .DSFNet import Unet_multistage
from .res2net import res2net50_v1b_26w_4s


upsample = lambda x, scale: F.interpolate(
    x, scale_factor=scale, mode='bilinear', align_corners=True)


class ConvReLU(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_sz,
            stride=1,
            relu=True,
            pd=True,
            bn=False):
        super(ConvReLU, self).__init__()
        padding = int((kernel_sz - 1) / 2) if pd else 0
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_sz, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch) if bn else None
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
        self.ft_conv = nn.Sequential(ConvReLU(ft_in, ft_out, 1))
        self.conv = nn.Sequential(
            ConvReLU(out_ch + ft_out, (out_ch + ft_out) // 2, 3),
            nn.Conv2d(
                (out_ch + ft_out) // 2, out_ch, 3, padding=1))

    def forward(self, ft_cur, ft_pre):
        ft_cur = self.ft_conv(ft_cur)
        return self.conv(torch.cat((ft_cur, ft_pre), dim=1))


class Hourglass(nn.Module):
    def __init__(self, input_ch, output_ch, ch=None):
        super(Hourglass, self).__init__()
        if ch is None:
            ch = [32, 32, 32, 32]
        ch = [input_ch] + ch
        self.encoder_1 = nn.Sequential(
            ConvReLU(ch[0], (ch[0] + ch[1]) // 2, 3),
            ConvReLU((ch[0] + ch[1]) // 2, ch[1], 3))
        self.encoder_2 = nn.Sequential(
            ConvReLU(ch[1], (ch[1] + ch[2]) // 2, 3),
            ConvReLU((ch[1] + ch[2]) // 2, ch[2], 3))
        self.encoder_3 = nn.Sequential(
            ConvReLU(ch[2], (ch[2] + ch[3]) // 2, 3),
            ConvReLU((ch[2] + ch[3]) // 2, ch[3], 3))
        self.encoder_4 = nn.Sequential(
            ConvReLU(ch[3], (ch[3] + ch[4]) // 2, 3),
            ConvReLU((ch[3] + ch[4]) // 2, ch[4], 3))
        self.encoder_5 = nn.Sequential(
            ConvReLU(ch[4], ch[4], 3),
            ConvReLU(ch[4], ch[4], 3))
        self.decoder_4 = nn.Sequential(
            ConvReLU(ch[4], (ch[3] + ch[4]) // 2, 3),
            ConvReLU((ch[3] + ch[4]) // 2, ch[3], 3))
        self.decoder_3 = nn.Sequential(
            ConvReLU(ch[3], (ch[2] + ch[3]) // 2, 3),
            ConvReLU((ch[2] + ch[3]) // 2, ch[2], 3))
        self.decoder_2 = nn.Sequential(
            ConvReLU(ch[2], (ch[1] + ch[2]) // 2, 3),
            ConvReLU((ch[1] + ch[2]) // 2, ch[1], 3))
        self.decoder_1 = nn.Sequential(
            ConvReLU(ch[1], (ch[0] + ch[1]) // 2, 3),
            ConvReLU((ch[0] + ch[1]) // 2, output_ch, 3))
        self.maxpool = nn.MaxPool2d(2, 2)
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        encoder_1 = self.encoder_1(x)
        encoder_2 = self.encoder_2(self.maxpool(encoder_1))
        encoder_3 = self.encoder_3(self.maxpool(encoder_2))
        encoder_4 = self.encoder_4(self.maxpool(encoder_3))
        encoder_5 = self.encoder_5(self.maxpool(encoder_4))
        decoder_4 = self.decoder_4(upsample(encoder_5, 2) + encoder_4)
        decoder_3 = self.decoder_3(upsample(decoder_4, 2) + encoder_3)
        decoder_2 = self.decoder_2(upsample(decoder_3, 2) + encoder_2)
        return self.decoder_1(upsample(decoder_2, 2) + encoder_1)


class Transformer(nn.Module):
    """Legacy trajectory sequence encoder retained for ablation runs."""

    def __init__(
            self, input_dim, embed_dim, num_heads, num_layers, output_dim):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer1 = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x, src_key_padding_mask):
        x = self.embedding(x)
        x = self.transformer1(
            x, src_key_padding_mask=src_key_padding_mask)
        return self.fc(x[:, -1, :])


class CrossAttentionLayer(nn.Module):
    """Unused legacy experiment module kept for checkpoint compatibility."""

    def __init__(self, dim_q, num_heads, dim_kv=None, dropout=0.0):
        super().__init__()
        dim_kv = dim_kv or dim_q
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_q,
            num_heads=num_heads,
            kdim=dim_kv,
            vdim=dim_kv,
            dropout=dropout,
            batch_first=True)
        self.ln_q = nn.LayerNorm(dim_q)
        self.ln_out = nn.LayerNorm(dim_q)
        self.ln_kv = nn.LayerNorm(dim_kv)

    def forward(self, q, k, v, key_padding_mask=None):
        q_residual = q
        q = self.ln_q(q)
        k = self.ln_kv(k)
        v = self.ln_kv(v)
        out, _ = self.attn(
            q, k, v, key_padding_mask=key_padding_mask)
        return self.ln_out(out + q_residual)


class TrajProjector(nn.Module):
    """Unused legacy experiment module kept for checkpoint compatibility."""

    def __init__(self, in_dim, out_dim, hidden=128, dropout=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


class RPNet(nn.Module):
    """VecRoad RPNet with optional road_self legacy trajectory extensions.

    With ``enable_trajectory_modules=False`` the registered modules, tensor
    resolutions, recursive anchor feedback, and state_dict keys follow the
    official VecRoad implementation. Trajectory-only parameters are created
    only for ``legacy_current`` runs.
    """

    def __init__(
            self,
            num_targets=4,
            backbone_pretrained=True,
            enable_trajectory_modules=False):
        super(RPNet, self).__init__()
        self.num_targets = num_targets
        self.enable_trajectory_modules = bool(enable_trajectory_modules)

        self.conv_2_side = ConvReLU(256, 128, 3, 1, bn=True)
        self.conv_3_side = ConvReLU(512, 128, 3, 1, bn=True)
        self.conv_4_side = ConvReLU(1024, 128, 3, 1, bn=True)
        self.conv_5_side = ConvReLU(2048, 128, 3, 1, bn=True)
        self.conv_fuse = ConvReLU(512, 128, 3, 1, bn=True)
        self.avgpool4 = nn.AvgPool2d(4, 4)

        self.road_seg = nn.Sequential(
            ConvReLU(128, 64, 3, 1, bn=True),
            ConvReLU(64, 64, 1, 1, bn=True))
        self.conv_road_final = nn.Conv2d(64, 1, 1, 1, 0)
        self.junc_seg = nn.Sequential(
            ConvReLU(128, 64, 3, 1, bn=True),
            ConvReLU(64, 64, 1, 1, bn=True))
        self.conv_junc_final = nn.Conv2d(64, 1, 1, 1, 0)
        self.fuse_module = Hourglass(
            128 + 64 + 64 + 32 * (self.num_targets - 1) + 1,
            32,
            [128, 128, 128, 128])

        self.ft_chs = [1024, 512, 256, 64]
        self.decoders = nn.ModuleList([
            DecoderBlock(self.ft_chs[0], 32, 32),
            DecoderBlock(self.ft_chs[1], 32, 32),
            DecoderBlock(self.ft_chs[2], 32, 32),
            DecoderBlock(self.ft_chs[3], 32, 32)])
        self.next_step_final = nn.Conv2d(32, 1, 1, 1, 0)
        self.conv_final = nn.Conv2d(32, 1, 3, 1, 1)

        if self.enable_trajectory_modules:
            self.transformer = Transformer(
                2, 32, num_heads=4, num_layers=1, output_dim=64)
            self.traj_to_img_fc = TrajProjector(
                in_dim=2, out_dim=128, hidden=128)
            self.cross_attention = CrossAttentionLayer(
                dim_q=128, num_heads=4, dim_kv=128, dropout=0.0)
            self.fuse_module_traj = Hourglass(
                128 + 64 + 64 + 64 + 32 * (self.num_targets - 1) + 1,
                32,
                [128, 128, 128, 128])
            self.upsample1 = nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=64,
                stride=64)

        self.init_weights()
        res2net = res2net50_v1b_26w_4s(pretrained=backbone_pretrained)
        self.stage_1 = nn.Sequential(res2net.conv1, res2net.relu)
        self.maxpool = res2net.maxpool
        self.stage_2 = res2net.layer1
        self.stage_3 = res2net.layer2
        self.stage_4 = res2net.layer3
        self.stage_5 = res2net.layer4

        if self.enable_trajectory_modules:
            self.stage_1_traj = nn.Sequential(
                nn.Conv2d(
                    1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                res2net.relu)
            self.stage_1_traj_aerial = nn.Sequential(
                nn.Conv2d(
                    4, 64, kernel_size=7, stride=2, padding=3, bias=False),
                res2net.relu)
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
            # Preserve the legacy checkpoint tensor shape. The placeholder is
            # explicitly resized to the official 1/4-resolution fusion map in
            # forward, so it does not restore the removed full-resolution path.
            self.missing_traj_feature = nn.Parameter(
                torch.zeros(1, 64, 256, 256))
            nn.init.normal_(
                self.missing_traj_feature, mean=0.0, std=0.02)

    def conv_stage(
            self,
            dim_in,
            dim_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True):
        return nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
            nn.Conv2d(
                dim_out,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU())

    def DSFupsample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(
                ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU())

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _forward_origin_backbone(self, aerial_image, feature_maps):
        stage_1 = self.stage_1(aerial_image)
        stage_2 = self.stage_2(self.maxpool(stage_1))
        stage_3 = self.stage_3(stage_2)
        stage_4 = self.stage_4(stage_3)
        stage_5 = self.stage_5(stage_4)

        stage_fuse = torch.cat([
            self.conv_2_side(stage_2),
            upsample(self.conv_3_side(stage_3), 2),
            upsample(self.conv_4_side(stage_4), 2),
            upsample(self.conv_5_side(stage_5), 2)], dim=1)
        stage_fuse = self.conv_fuse(stage_fuse)
        road_fts = self.road_seg(stage_fuse)
        road_final = self.conv_road_final(road_fts)
        junc_fts = self.junc_seg(stage_fuse)
        junc_final = self.conv_junc_final(junc_fts)

        feature_maps.update({
            'stage_1': stage_1,
            'stage_2': stage_2,
            'stage_3': stage_3,
            'stage_4': stage_4,
            'stage_5': stage_5,
            'stage_fuse': stage_fuse,
            'road_fts': road_fts,
            'junc_fts': junc_fts})
        return (
            stage_1,
            stage_2,
            stage_3,
            stage_4,
            stage_fuse,
            road_fts,
            junc_fts,
            road_final,
            junc_final)

    def forward(
            self,
            aerial_image,
            traj_image,
            aerial_traj_image,
            neighborhood_trajectory_norm,
            valid_mask,
            walked_path,
            NUM_TARGETS=None,
            test=False,
            model=None,
            use_traj=None):
        model = model or 'origin'
        use_traj = bool(use_traj)
        if use_traj and not self.enable_trajectory_modules:
            raise RuntimeError(
                'Trajectory input was requested, but RPNet was constructed '
                'with enable_trajectory_modules=False.')

        feature_maps = {}
        traj_final = None
        if model == 'origin':
            (
                stage_1,
                stage_2,
                stage_3,
                stage_4,
                stage_fuse,
                road_fts,
                junc_fts,
                road_final,
                junc_final,
            ) = self._forward_origin_backbone(aerial_image, feature_maps)
        elif model == 'DSFNet':
            if not self.enable_trajectory_modules:
                raise RuntimeError(
                    'DSFNet requires enable_trajectory_modules=True.')
            (
                stage_1_DFS,
                stage_2_DFS,
                stage_3_DFS,
                stage_4_DFS,
                stage_fuse,
                road_fts,
                junc_fts,
                road_final,
                junc_final,
                _traj_fts,
                traj_final,
                fi_ca1,
            ) = self.DSF(traj_image, aerial_image)
        else:
            raise ValueError('Unknown model mode: {!r}'.format(model))

        if test:
            if model == 'origin':
                return {
                    'road': upsample(road_final, 4),
                    'junc': upsample(junc_final, 4)}
            return {'road': road_final, 'junc': junc_final}

        next_points_placeholder = torch.zeros(
            (
                stage_fuse.shape[0],
                32 * (self.num_targets - 1),
                stage_fuse.shape[2],
                stage_fuse.shape[3]),
            device=stage_fuse.device,
            dtype=stage_fuse.dtype)

        if use_traj:
            have_trajectory = bool(
                neighborhood_trajectory_norm is not None
                and not torch.all(neighborhood_trajectory_norm == 0))
            if have_trajectory:
                neighborhood_trajectory = neighborhood_trajectory_norm.view(
                    neighborhood_trajectory_norm.size(0),
                    -1,
                    neighborhood_trajectory_norm.size(3))
                padding_mask = (~valid_mask).view(
                    neighborhood_trajectory_norm.size(0), -1)
                valid_traj_feature = self.transformer(
                    neighborhood_trajectory,
                    src_key_padding_mask=padding_mask)
                valid_traj_feature = self.upsample1(
                    valid_traj_feature.unsqueeze(2).unsqueeze(3))
                valid_traj_feature = F.interpolate(
                    valid_traj_feature,
                    size=stage_fuse.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
            else:
                valid_traj_feature = self.missing_traj_feature.expand(
                    aerial_image.size(0), -1, -1, -1)
                valid_traj_feature = F.interpolate(
                    valid_traj_feature,
                    size=stage_fuse.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
            feature_maps['valid_traj_feature'] = valid_traj_feature
            stage_fuse = torch.cat([
                stage_fuse,
                road_fts,
                junc_fts,
                walked_path,
                valid_traj_feature,
                next_points_placeholder], dim=1)
        else:
            stage_fuse = torch.cat([
                stage_fuse,
                road_fts,
                junc_fts,
                walked_path,
                next_points_placeholder], dim=1)

        if self.training:
            stage_fuse_list = [stage_fuse]

        anchor_fts = None
        next_points = []
        next_points_lowrs = []
        num_targets = (
            NUM_TARGETS if NUM_TARGETS is not None else self.num_targets)
        for index in range(num_targets):
            if self.training:
                fuse_input = stage_fuse_list[index]
            else:
                fuse_input = stage_fuse
            if use_traj:
                next_step = self.fuse_module_traj(fuse_input)
            else:
                next_step = self.fuse_module(fuse_input)
            next_points_lowrs.append(
                upsample(self.next_step_final(next_step), 4))

            if model == 'origin':
                decoded_ft_4 = self.decoders[0](
                    upsample(stage_4, 2), next_step)
                decoded_ft_3 = self.decoders[1](
                    upsample(stage_3, 2), decoded_ft_4)
                decoded_ft_2 = self.decoders[2](
                    upsample(stage_2, 2), upsample(decoded_ft_3, 2))
                decoded_ft_1 = self.decoders[3](
                    upsample(stage_1, 2), upsample(decoded_ft_2, 2))
            else:
                decoded_ft_4 = self.up4_anchor(torch.cat(
                    (self.trans4_anchor(fi_ca1), stage_4_DFS), 1))
                decoded_ft_3 = self.up3_anchor(torch.cat(
                    (self.trans3_anchor(decoded_ft_4), stage_3_DFS), 1))
                decoded_ft_2 = self.up2_anchor(torch.cat(
                    (self.trans2_anchor(decoded_ft_3), stage_2_DFS), 1))
                decoded_ft_1 = self.up1_anchor(torch.cat(
                    (self.trans1_anchor(decoded_ft_2), stage_1_DFS), 1))
                decoded_ft_1 = self.up0_anchor(decoded_ft_1)

            feature_maps['decoded_ft_4_step_{}'.format(index)] = decoded_ft_4
            feature_maps['decoded_ft_3_step_{}'.format(index)] = decoded_ft_3
            feature_maps['decoded_ft_2_step_{}'.format(index)] = decoded_ft_2
            feature_maps['decoded_ft_1_step_{}'.format(index)] = decoded_ft_1

            channel_index = -(self.num_targets - index - 1) * 32
            if index < self.num_targets - 1:
                pooled_anchor = self.avgpool4(decoded_ft_1)
                if anchor_fts is None:
                    anchor_fts = pooled_anchor
                else:
                    anchor_fts += pooled_anchor
                if self.training:
                    stage_fuse_list.append(stage_fuse_list[index].clone())
                    stage_fuse_list[index + 1][
                        :,
                        channel_index:channel_index + 32
                        if channel_index + 32 != 0 else None,
                        :,
                        :,
                    ] = anchor_fts
                else:
                    stage_fuse[
                        :,
                        channel_index:channel_index + 32
                        if channel_index + 32 != 0 else None,
                        :,
                        :,
                    ] = anchor_fts
                feature_maps[
                    'anchor_fts_step_{}'.format(index)] = anchor_fts

            next_points.append(self.conv_final(decoded_ft_1))

        return {
            'road': road_final,
            'junc': junc_final,
            'anchor': torch.cat(next_points, dim=1),
            'anchor_lowrs': torch.cat(next_points_lowrs, dim=1),
            'traj_road': traj_final,
            'feature_maps': feature_maps}


def build_model(
        num_targets=4,
        backbone_pretrained=True,
        enable_trajectory_modules=False):
    return RPNet(
        num_targets=num_targets,
        backbone_pretrained=backbone_pretrained,
        enable_trajectory_modules=enable_trajectory_modules)


if __name__ == '__main__':
    model = build_model(backbone_pretrained=False).eval()
    batch_size = 1
    input_img = torch.zeros((batch_size, 3, 256, 256))
    input_walked_path = torch.zeros((batch_size, 1, 64, 64))
    with torch.no_grad():
        output = model(
            input_img,
            None,
            None,
            None,
            None,
            input_walked_path,
            model='origin',
            use_traj=False)
    print({key: tuple(value.shape) for key, value in output.items()
           if torch.is_tensor(value)})
