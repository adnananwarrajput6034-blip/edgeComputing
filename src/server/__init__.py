"""
Server Module - Runs on Cloud/Laptop
=====================================

This module contains all code that executes on the central server.
The server role differs by strategy:

Strategy A (Centralized):
    - Receives raw audio + labels
    - Performs STFT conversion
    - Trains model centrally
    - Broadcasts updated model

Strategy B (Hybrid):
    - Receives spectrograms + labels
    - Trains model centrally
    - Broadcasts updated model

Strategy C (Federated):
    - Receives model weights from clients
    - Performs FedAvg aggregation (hand-rolled, over MQTT)
    - Broadcasts global model

Submodules:
    - training: model definitions (audio_cnn, fusion_model)
    - aggregation: FedAvg weight averaging (Strategy C)
    - api: reserved

The actual server entry points live in src/experiments/strategy_{a,b,c}_server.py
and are spawned by scripts/run_strategy.py — see docs/CODE_FLOW.md.
"""

# Submodules are NOT eagerly imported — TF should only load when the user
# actually reaches for those features. Import explicitly, e.g.:
#     from src.server.aggregation.fedavg import FedAvgAggregator
__all__ = []
