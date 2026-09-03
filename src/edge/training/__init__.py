"""
Local Training Module (for Strategy C - Federated)
===================================================

This module handles on-device training for the federated learning strategy.

On-device training challenges:
1. Limited RAM (8GB shared with OS and other processes)
2. Limited compute (4 ARM cores, no GPU)
3. Heat management (throttling under sustained load)
4. Power constraints (battery if mobile)

Solutions implemented:
1. Gradient accumulation (simulate larger batches)
2. TFLite for inference, full TF for training
3. Memory-mapped datasets
4. Training scheduling (avoid peak usage times)

Classes:
    LocalDataset: Handles local data buffering
"""

from .dataset import LocalDataset

__all__ = ["LocalDataset"]
