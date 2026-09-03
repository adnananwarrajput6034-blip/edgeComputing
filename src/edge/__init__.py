"""
Edge Module - Runs on Raspberry Pi
===================================

This module contains all code that executes on the Raspberry Pi edge devices.
It handles sensor capture, processing, strategy execution, and communication
with the central server.

Submodules:
    - sensors: Camera and microphone capture with synchronization
    - processing: STFT, YOLOv8, and feature extraction
    - strategies: Implementation of A (Centralized), B (Hybrid), C (Federated)
    - training: Local training for Strategy C
    - communication: MQTT and Flower client
    - metrics: Power, bandwidth, latency measurement

Usage:
    python -m src.edge.main --node-id A --server-ip 192.168.1.100
"""


__all__ = []
