# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# from turtle import shape
import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from einops import rearrange
from .positional_encoder import PosCNN
from .common import LayerNorm2d
from .transformer import Attention


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        multimask_output = False,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        # self.positional_encoder = PosCNN(in_chans=256, embed_dim=256)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.multimask_output = multimask_output

        # --------------------- requires_grad -------------------------------
        for n, value in self.image_encoder.named_parameters():
          if "cnn_embed" not in n and "pos_post_embed" not in n and "Adapter" not in n and "2.attn.rel_pos" not in n \
              and "5.attn.rel_pos" not in n and "8.attn.rel_pos" not in n and "11.attn.rel_pos" not in n \
              and "upneck" not in n: # and "2.mlp" not in n and "5.mlp" not in n and "8.mlp" not in n and "11.mlp" not in n:
            value.requires_grad = False
        # for n, value in self.image_encoder.named_parameters():
        #   if "Adapter" not in n and "pos_post_embed" not in n and "rel_pos" not in n: 
        #     value.requires_grad = False
            
        # -------------------------------------------------------------------

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self, 
        imgs: torch.Tensor,
        points: Tuple[torch.Tensor, torch.Tensor], 
        points_label: torch.Tensor,  
    ) -> torch.Tensor:

        imgs = imgs.permute(0, 3, 1, 2)
        imgs = (imgs - self.pixel_mean) / self.pixel_std
        B, C, H, W = imgs.shape
        image_embeddings = self.image_encoder(imgs) #B, 16, 16, 256

        points = (points, points_label)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )         
        # get_dense_pe = self.positional_encoder(image_embeddings)
        dense_embeddings = dense_embeddings.expand(B, -1, image_embeddings.shape[2], image_embeddings.shape[3])
        low_res_masks = self.mask_decoder( 
                    image_embeddings=image_embeddings,
                    # image_pe=get_dense_pe, 
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings, 
                    multimask_output=self.multimask_output,
                    )
        masks = F.interpolate(low_res_masks, (H, W), mode="bilinear", align_corners=False)

        return masks


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x