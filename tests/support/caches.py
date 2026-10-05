"""Tests for caches."""

import torch


class _OutputCache:
    def __init__(self, length):
        self.layers = [(
            torch.zeros((1, 1, length, 2)),
            torch.zeros((1, 1, length, 2)),
        )]

    def get_seq_length(self):
        return self.layers[0][0].shape[-2]

    def update(self, key, value, _layer_index):
        old_key, old_value = self.layers[0]
        self.layers[0] = (
            torch.cat([old_key, key], dim=-2),
            torch.cat([old_value, value], dim=-2),
        )

    def crop(self, length):
        key, value = self.layers[0]
        self.layers[0] = (key[..., :length, :], value[..., :length, :])
