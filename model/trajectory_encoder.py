"""Independent temporal encoder for Stage 1C trajectory fragments."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


class TrajectoryFragmentEncoder(nn.Module):
    """Encode each ordered trajectory fragment without fragment interaction.

    The fragment dimension is folded into the Transformer batch dimension,
    never into its sequence dimension. Consequently, attention is restricted
    to points from the same fragment.
    """

    INPUT_FEATURE_DIM = 7
    REQUIRED_BATCH_KEYS = (
        "traj_xy_norm",
        "traj_time_delta",
        "point_mask",
        "fragment_mask",
        "point_inside_mask",
        "segment_only",
    )

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.hidden_dim = int(hidden_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(self.INPUT_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.embedding_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
        self.cls_token = nn.Parameter(
            torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    @staticmethod
    def _sinusoidal_position_encoding(
        sequence_length: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create a dynamic position encoding with no fixed length limit."""

        if sequence_length == 0:
            return torch.zeros(
                (0, hidden_dim), device=device, dtype=dtype)
        positions = torch.arange(
            sequence_length,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(
                0,
                hidden_dim,
                2,
                device=device,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / hidden_dim)
        )
        angles = positions * frequencies.unsqueeze(0)
        encoding = torch.zeros(
            (sequence_length, hidden_dim),
            device=device,
            dtype=torch.float32,
        )
        encoding[:, 0::2] = torch.sin(angles)
        odd_width = encoding[:, 1::2].shape[1]
        if odd_width > 0:
            encoding[:, 1::2] = torch.cos(angles[:, :odd_width])
        return encoding.to(dtype=dtype)

    @classmethod
    def _validate_batch(
        cls,
        trajectory_batch: Dict[str, torch.Tensor],
    ) -> None:
        missing = [
            key
            for key in cls.REQUIRED_BATCH_KEYS
            if key not in trajectory_batch
        ]
        if missing:
            raise KeyError(
                "trajectory batch is missing: {}".format(
                    ", ".join(missing)))
        for key in cls.REQUIRED_BATCH_KEYS:
            if not torch.is_tensor(trajectory_batch[key]):
                raise TypeError(
                    "trajectory batch field {!r} must be a tensor".format(
                        key))

        xy_norm = trajectory_batch["traj_xy_norm"]
        if xy_norm.ndim != 4 or xy_norm.shape[-1] != 2:
            raise ValueError(
                "traj_xy_norm must have shape [B, N, T, 2]")
        batch_size, fragment_count, point_count, _ = xy_norm.shape
        point_shape = (batch_size, fragment_count, point_count)
        fragment_shape = (batch_size, fragment_count)
        for key in (
            "traj_time_delta",
            "point_mask",
            "point_inside_mask",
        ):
            if tuple(trajectory_batch[key].shape) != point_shape:
                raise ValueError(
                    "{} must have shape [B, N, T]".format(key))
        for key in ("fragment_mask", "segment_only"):
            if tuple(trajectory_batch[key].shape) != fragment_shape:
                raise ValueError(
                    "{} must have shape [B, N]".format(key))

        devices = {
            trajectory_batch[key].device
            for key in cls.REQUIRED_BATCH_KEYS
        }
        if len(devices) != 1:
            raise ValueError(
                "all trajectory batch fields must share one device")

    @staticmethod
    def _motion_features(
        xy_norm: torch.Tensor,
        time_delta: torch.Tensor,
        point_mask: torch.Tensor,
    ):
        """Return ordered displacement and signed log time-gap features."""

        delta_xy = torch.zeros_like(xy_norm)
        delta_time = torch.zeros_like(time_delta)
        if xy_norm.shape[2] > 1:
            consecutive_mask = (
                point_mask[:, :, 1:]
                & point_mask[:, :, :-1]
            )
            delta_xy[:, :, 1:] = (
                xy_norm[:, :, 1:] - xy_norm[:, :, :-1]
            ) * consecutive_mask.unsqueeze(-1).to(dtype=xy_norm.dtype)
            delta_time[:, :, 1:] = (
                time_delta[:, :, 1:] - time_delta[:, :, :-1]
            ) * consecutive_mask.to(dtype=time_delta.dtype)
        delta_time = torch.sign(delta_time) * torch.log1p(
            torch.abs(delta_time))
        return delta_xy, delta_time

    def forward(
        self,
        trajectory_batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        self._validate_batch(trajectory_batch)
        xy_norm_input = trajectory_batch["traj_xy_norm"]
        time_delta_input = trajectory_batch["traj_time_delta"]
        point_mask_output = trajectory_batch["point_mask"]
        fragment_mask_output = trajectory_batch["fragment_mask"]
        point_inside_input = trajectory_batch["point_inside_mask"]
        segment_only_input = trajectory_batch["segment_only"]

        parameter = self.cls_token
        if xy_norm_input.device != parameter.device:
            raise ValueError(
                "trajectory batch and encoder must be on the same device")
        feature_dtype = parameter.dtype
        xy_norm = xy_norm_input.to(dtype=feature_dtype)
        time_delta = time_delta_input.to(dtype=feature_dtype)
        point_mask = point_mask_output.to(dtype=torch.bool)
        fragment_mask = fragment_mask_output.to(dtype=torch.bool)
        point_inside = point_inside_input.to(dtype=feature_dtype)
        segment_only = segment_only_input.to(dtype=feature_dtype)

        batch_size, fragment_count, point_count, _ = xy_norm.shape
        output_shape = (
            batch_size,
            fragment_count,
            point_count,
            self.hidden_dim,
        )
        point_tokens = torch.zeros(
            output_shape,
            device=xy_norm.device,
            dtype=feature_dtype,
        )
        fragment_tokens = torch.zeros(
            (batch_size, fragment_count, self.hidden_dim),
            device=xy_norm.device,
            dtype=feature_dtype,
        )
        if batch_size == 0 or fragment_count == 0:
            return {
                "point_tokens": point_tokens,
                "fragment_tokens": fragment_tokens,
                "point_mask": point_mask_output,
                "fragment_mask": fragment_mask_output,
            }

        effective_point_mask = (
            point_mask & fragment_mask.unsqueeze(-1))
        delta_xy, delta_time = self._motion_features(
            xy_norm,
            time_delta,
            effective_point_mask,
        )
        segment_feature = segment_only.unsqueeze(-1).expand(
            batch_size, fragment_count, point_count)
        input_features = torch.cat(
            (
                xy_norm,
                delta_xy,
                delta_time.unsqueeze(-1),
                point_inside.unsqueeze(-1),
                segment_feature.unsqueeze(-1),
            ),
            dim=-1,
        )
        input_features = input_features * effective_point_mask.unsqueeze(
            -1).to(dtype=feature_dtype)
        point_embeddings = self.input_projection(input_features)

        flat_count = batch_size * fragment_count
        flat_embeddings = point_embeddings.reshape(
            flat_count, point_count, self.hidden_dim)
        flat_point_mask = effective_point_mask.reshape(
            flat_count, point_count)
        valid_fragment_mask = (
            fragment_mask.reshape(flat_count)
            & flat_point_mask.any(dim=1)
        )
        valid_indices = torch.nonzero(
            valid_fragment_mask, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return {
                "point_tokens": point_tokens,
                "fragment_tokens": fragment_tokens,
                "point_mask": point_mask_output,
                "fragment_mask": fragment_mask_output,
            }

        valid_embeddings = flat_embeddings.index_select(
            0, valid_indices)
        valid_point_mask = flat_point_mask.index_select(
            0, valid_indices)
        valid_count = valid_embeddings.shape[0]
        cls_tokens = self.cls_token.expand(valid_count, -1, -1)
        sequence = torch.cat((cls_tokens, valid_embeddings), dim=1)
        position_encoding = self._sinusoidal_position_encoding(
            point_count + 1,
            self.hidden_dim,
            sequence.device,
            sequence.dtype,
        )
        sequence = self.embedding_dropout(
            sequence + position_encoding.unsqueeze(0))
        key_padding_mask = torch.cat(
            (
                torch.zeros(
                    (valid_count, 1),
                    dtype=torch.bool,
                    device=sequence.device,
                ),
                ~valid_point_mask,
            ),
            dim=1,
        )
        encoded = self.temporal_encoder(
            sequence,
            src_key_padding_mask=key_padding_mask,
        )
        encoded_points = encoded[:, 1:] * valid_point_mask.unsqueeze(
            -1).to(dtype=encoded.dtype)

        flat_point_tokens = point_tokens.reshape(
            flat_count, point_count, self.hidden_dim)
        flat_fragment_tokens = fragment_tokens.reshape(
            flat_count, self.hidden_dim)
        flat_point_tokens = flat_point_tokens.index_copy(
            0, valid_indices, encoded_points)
        flat_fragment_tokens = flat_fragment_tokens.index_copy(
            0, valid_indices, encoded[:, 0])
        point_tokens = flat_point_tokens.reshape(output_shape)
        fragment_tokens = flat_fragment_tokens.reshape(
            batch_size, fragment_count, self.hidden_dim)

        return {
            "point_tokens": point_tokens,
            "fragment_tokens": fragment_tokens,
            "point_mask": point_mask_output,
            "fragment_mask": fragment_mask_output,
        }
