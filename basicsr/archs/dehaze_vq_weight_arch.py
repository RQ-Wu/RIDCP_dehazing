import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv

from .network_swinir import RSTB
from .ridcp_utils import ResBlock, CombineQuantBlock 
from .vgg_arch import VGGFeatureExtractor


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.
    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.
    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')
        
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, weight_path='pretrained_models/weight_for_matching_dehazing_Flickr.pth', beta=0.25, LQ_stage=False, use_weight=True, weight_alpha=1.0):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta
        self.use_weight = use_weight
        self.weight_alpha = weight_alpha
        if self.use_weight:
            self.weight = nn.Parameter(torch.load(weight_path))
            self.weight.requires_grad = False 
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
    
    def dist(self, x, y):
        if x.shape == y.shape:
            return (x - y) ** 2
        else:
            return torch.sum(x ** 2, dim=1, keepdim=True) + \
                    torch.sum(y**2, dim=1) - 2 * \
                    torch.matmul(x, y.t())
    
    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h*w, c)
        y = y.reshape(b, h*w, c)

        gmx = x.transpose(1, 2) @ x / (h*w)
        gmy = y.transpose(1, 2) @ y / (h*w)
    
        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, current_iter=None, weight_alpha=None):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization. 
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight

        d = self.dist(z_flattened, codebook)
        if self.use_weight and self.LQ_stage:
            if weight_alpha is not None:
                self.weight_alpha = weight_alpha
            d = d * torch.exp(self.weight_alpha * self.weight)
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)

            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z)**2)
        q_latent_loss = torch.mean((z_q - z.detach())**2)

        if self.LQ_stage and gt_indices is not None:
            # codebook_loss = self.dist(z_q, z_q_gt.detach()).mean() \
                            # + self.beta * self.dist(z_q_gt.detach(), z) 
            codebook_loss = self.beta * self.dist(z_q_gt.detach(), z) 
            texture_loss = self.gram_loss(z, z_q_gt.detach()) 
            # print("codebook loss:", codebook_loss.mean(), "\ntexture_loss: ", texture_loss.mean())
            codebook_loss = codebook_loss + texture_loss 
        else:
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])
    
    def get_codebook_entry(self, indices):
        b, _, h, w = indices.shape

        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)        
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q

class SwinLayers(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256, 
                blk_depth=6,
                num_heads=8,
                window_size=8,
                **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(4):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w) 
        return x


class MultiScaleEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 LQ_stage=True,
                 **swin_opts,
                 ):
        super().__init__()
        self.LQ_stage = LQ_stage
        ksz = 3

        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

        if LQ_stage: 
            self.blocks.append(SwinLayers(**swin_opts))

    def forward(self, input):
        # input.requires_grad = True
        x = self.in_conv(input)
        # if self.LQ_stage:
        #     print('input: ', input.requires_grad)
        #     for p in self.in_conv.parameters():
        #         print('conv: ', p.requires_grad)
        #     print('first output:', x.requires_grad)

        for idx, m in enumerate(self.blocks):
            with torch.backends.cudnn.flags(enabled=False):
                x = m(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()

        self.block = []
        self.block += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
        ]

        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)

class WarpBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.offset = nn.Conv2d(in_channel * 2, in_channel, 3, stride=1, padding=1)
        self.dcn = DCNv2Pack(in_channel, in_channel, 3, padding=1, deformable_groups=4)

    def forward(self, x_vq, x_residual):
        x_residual = self.offset(torch.cat([x_vq, x_residual], dim=1))
        feat_after_warp = self.dcn(x_vq, x_residual)

        return feat_after_warp


class MultiScaleDecoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 only_residual=False,
                 use_warp=True
                 ):
        super().__init__()
        self.only_residual = only_residual
        self.use_warp = use_warp
        self.upsampler = nn.ModuleList()
        self.warp = nn.ModuleList()
        res =  input_res // (2 ** max_depth)
        for i in range(max_depth):
            in_channel, out_channel = channel_query_dict[res], channel_query_dict[res * 2]
            self.upsampler.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                ResBlock(out_channel, out_channel, norm_type, act_type),
                ResBlock(out_channel, out_channel, norm_type, act_type),
                )
            )
            self.warp.append(WarpBlock(out_channel))
            res = res * 2

    def forward(self, input, code_decoder_output):
        x = input
        for idx, m in enumerate(self.upsampler):
            with torch.backends.cudnn.flags(enabled=False):
                if not self.only_residual:
                    x = m(x)
                    if self.use_warp:
                        x_vq = self.warp[idx](code_decoder_output[idx], x)
                        # print(idx, x.mean(), x_vq.mean())
                        x = x + x_vq * (x.mean() / x_vq.mean())
                    else:
                        x = x + code_decoder_output[idx]
                else:
                    x = m(x)
        # print()
        return x

@ARCH_REGISTRY.register()
class VQWeightDehazeNet(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 use_semantic_loss=False,
                 use_residual=True,
                 only_residual=False,
                 use_weight=False,
                 use_warp=True,
                 weight_alpha=1.0,
                 **ignore_kwargs):
        super().__init__()

        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[:, 0]
        codebook_emb_num = codebook_params[:, 1].astype(int)
        codebook_emb_dim = codebook_params[:, 2].astype(int)

        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.use_residual = use_residual
        self.only_residual = only_residual
        self.use_weight = use_weight
        self.use_warp = use_warp
        self.weight_alpha = weight_alpha

        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }

        # build encoder 
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))
        self.multiscale_encoder = MultiScaleEncoder(
                                in_channel,     
                                self.max_depth,  
                                self.gt_res, 
                                channel_query_dict,
                                norm_type, act_type, LQ_stage
                            )
        if self.LQ_stage and self.use_residual:
            self.multiscale_decoder = MultiScaleDecoder(
                                in_channel,     
                                self.max_depth,  
                                self.gt_res, 
                                channel_query_dict,
                                norm_type, act_type, only_residual, use_warp=self.use_warp
            )


        # build decoder
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2**self.max_depth * 2**i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)
        self.residual_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        # build multi-scale vector quantizers 
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()

        for scale in range(0, codebook_params.shape[0]):
            quantize = VectorQuantizer(
                codebook_emb_num[scale],
                codebook_emb_dim[scale],
                LQ_stage=self.LQ_stage,
                use_weight=self.use_weight,
                weight_alpha=self.weight_alpha
            )
            self.quantize_group.append(quantize)

            scale_in_ch = channel_query_dict[self.codebook_scale[scale]]
            if scale == 0:
                quant_conv_in_ch = scale_in_ch
                comb_quant_in_ch1 = codebook_emb_dim[scale]
                comb_quant_in_ch2 = 0
            else:
                quant_conv_in_ch = scale_in_ch * 2
                comb_quant_in_ch1 = codebook_emb_dim[scale - 1]
                comb_quant_in_ch2 = codebook_emb_dim[scale]

            self.before_quant_group.append(nn.Conv2d(quant_conv_in_ch, codebook_emb_dim[scale], 1))
            self.after_quant_group.append(CombineQuantBlock(comb_quant_in_ch1, comb_quant_in_ch2, scale_in_ch))

        # semantic loss for HQ pretrain stage
        self.use_semantic_loss = use_semantic_loss
        if use_semantic_loss:
            self.conv_semantic = nn.Sequential(
                nn.Conv2d(512, 512, 1, 1, 0),
                nn.ReLU(),
                )
            self.vgg_feat_layer = 'relu4_4'
            self.vgg_feat_extractor = VGGFeatureExtractor([self.vgg_feat_layer]) 

    def encode_and_decode(self, input, gt_indices=None, current_iter=None, weight_alpha=None):
        # if self.training:
        #     for p in self.multiscale_encoder.parameters():
        #         p.requires_grad = True
        enc_feats = self.multiscale_encoder(input)

        if self.use_semantic_loss:
            with torch.no_grad():
                vgg_feat = self.vgg_feat_extractor(input)[self.vgg_feat_layer]

        codebook_loss_list = []
        indices_list = []
        semantic_loss_list = []
        code_decoder_output = []

        quant_idx = 0
        prev_dec_feat = None
        prev_quant_feat = None
        out_img = None
        out_img_residual = None

        x = enc_feats
        for i in range(self.max_depth):
            cur_res = self.gt_res // 2**self.max_depth * 2**i
            if cur_res in self.codebook_scale:  # needs to perform quantize
                if prev_dec_feat is not None:
                    before_quant_feat = torch.cat((x, prev_dec_feat), dim=1)
                else:
                    before_quant_feat = x
                feat_to_quant = self.before_quant_group[quant_idx](before_quant_feat)

                if weight_alpha is not None:
                    self.weight_alpha = weight_alpha
                if gt_indices is not None:
                    z_quant, codebook_loss, indices = self.quantize_group[quant_idx](feat_to_quant, gt_indices[quant_idx], weight_alpha=self.weight_alpha)
                else:
                    z_quant, codebook_loss, indices = self.quantize_group[quant_idx](feat_to_quant, weight_alpha=self.weight_alpha)

                if self.use_semantic_loss:
                    semantic_z_quant = self.conv_semantic(z_quant)
                    semantic_loss = F.mse_loss(semantic_z_quant, vgg_feat)
                    semantic_loss_list.append(semantic_loss)
                
                if not self.use_quantize:
                    z_quant = feat_to_quant

                after_quant_feat = self.after_quant_group[quant_idx](z_quant, prev_quant_feat)

                codebook_loss_list.append(codebook_loss)
                indices_list.append(indices)

                quant_idx += 1
                prev_quant_feat = z_quant
                x = after_quant_feat

            x = self.decoder_group[i](x)
            code_decoder_output.append(x)
            prev_dec_feat = x
        
        out_img = self.out_conv(x)

        if self.LQ_stage and self.use_residual:
            if self.only_residual:
                residual_feature = self.multiscale_decoder(enc_feats, code_decoder_output)
            else:
                residual_feature = self.multiscale_decoder(enc_feats.detach(), code_decoder_output)
            out_img_residual = self.residual_conv(residual_feature)

        if len(codebook_loss_list) > 0:
            codebook_loss = sum(codebook_loss_list) 
        else:
            codebook_loss = 0
        semantic_loss = sum(semantic_loss_list) if len(semantic_loss_list) else codebook_loss * 0

        return out_img, out_img_residual, codebook_loss, semantic_loss, feat_to_quant, z_quant, indices_list
    
    def decode_indices(self, indices):
        assert len(indices.shape) == 4, f'shape of indices must be (b, 1, h, w), but got {indices.shape}'

        z_quant = self.quantize_group[0].get_codebook_entry(indices)
        x = self.after_quant_group[0](z_quant)

        for m in self.decoder_group:
            x = m(x)
        out_img = self.out_conv(x)
        return out_img

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height
        output_width = width
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x
                output_end_x = input_end_x
                output_start_y = input_start_y
                output_end_y = input_end_y

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad)
                output_end_x_tile = output_start_x_tile + input_tile_width
                output_start_y_tile = (input_start_y - input_start_y_pad)
                output_end_y_tile = output_start_y_tile + input_tile_height

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                       output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                  output_start_x_tile:output_end_x_tile]
        return output

    @torch.no_grad()
    def test(self, input, weight_alpha=None):
        org_use_semantic_loss = self.use_semantic_loss
        self.use_semantic_loss = False

        # padding to multiple of window_size * 8
        wsz = 32
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]

        output_vq, output, _, _, _, after_quant, index = self.encode_and_decode(input, None, None, weight_alpha=weight_alpha)

        if output is not None:
            output = output[..., :h_old, :w_old]
        if output_vq is not None:
            output_vq = output_vq[..., :h_old, :w_old]

        self.use_semantic_loss = org_use_semantic_loss 
        return output, index

    def forward(self, input, gt_indices=None, weight_alpha=None):

        if gt_indices is not None:
            # in LQ training stage, need to pass GT indices for supervise.
            dec, dec_residual, codebook_loss, semantic_loss, quant_before_feature, quant_after_feature, indices = self.encode_and_decode(input, gt_indices, weight_alpha=weight_alpha)
        else:
            # in HQ stage, or LQ test stage, no GT indices needed.
            dec, dec_residual, codebook_loss, semantic_loss, quant_before_feature, quant_after_feature, indices = self.encode_and_decode(input, weight_alpha=weight_alpha)

        return dec, dec_residual, codebook_loss, semantic_loss, quant_before_feature, quant_after_feature, indices
