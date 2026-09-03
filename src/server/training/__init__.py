"""
Server Training Module
======================

Handles centralized training for Strategy A and B.

In centralized training:
1. Server receives data from edge nodes (via MQTT)
2. For Strategy A: Raw audio + labels → STFT → Training
3. For Strategy B: Spectrograms + labels → Training
4. Model is updated and broadcast to edge nodes

Classes:
    DataReceiver: Handles incoming data from edges
    ModelBroadcaster: Sends updated model to edges

Submodules:
    models/: Neural network architectures
"""


__all__ = []
