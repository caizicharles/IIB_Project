import torch


def custom_collate_fn(batch):
  keys = batch[0].keys()
  return {key: torch.stack([item[key] for item in batch]) for key in keys}