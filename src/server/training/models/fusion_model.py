"""
Fusion Model
============

Multimodal fusion model combining vision and audio branches.

Architecture:
    Vision Branch: MobileNetV3-Small (pre-trained, frozen)
        Input: (224, 224, 3) RGB image
        Output: 256-dim feature vector

    Audio Branch: Custom CNN (trained from scratch)
        Input: (51, 128, 1) mel-spectrogram
        Output: 128-dim feature vector

    Fusion Head:
        Concatenate → Dense(256) → Dropout → Dense(num_classes)

Total: ~1.33M parameters (~390K trainable)
    - MobileNetV3-Small: ~940K (frozen, ImageNet weights)
    - Audio CNN: ~111K (trainable)
    - Fusion head: ~280K (trainable)

Modality Dropout:
    During training, randomly drop audio or vision features
    to make model robust to single-modality scenarios.
"""

import logging
from typing import Tuple, Optional

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None


def FusionModel(
    num_classes: int = 10,
    vision_trainable: bool = False,
    audio_feature_dim: int = 128,
    vision_feature_dim: int = 256,
    dropout_rate: float = 0.3,
    modality_dropout: float = 0.1
):
    """
    Create multimodal fusion model.

    Args:
        num_classes: Number of output classes
        vision_trainable: Whether to fine-tune vision backbone
        audio_feature_dim: Dimension of audio features
        vision_feature_dim: Dimension of vision features
        dropout_rate: Classification head dropout
        modality_dropout: Probability of dropping a modality

    Returns:
        Compiled Keras model with two inputs
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available")

    # ============ Vision Branch ============
    # Input: RGB image
    vision_input = layers.Input(shape=(224, 224, 3), name='vision_input')

    # MobileNetV3-Small backbone
    mobilenet = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    mobilenet.trainable = vision_trainable

    # Vision feature extraction
    vision_features = mobilenet(vision_input)
    vision_features = layers.Dense(
        vision_feature_dim,
        activation='relu',
        name='vision_projection'
    )(vision_features)

    # ============ Audio Branch ============
    # Input: Mel-spectrogram
    audio_input = layers.Input(shape=(51, 128, 1), name='audio_input')

    # Audio CNN
    x = layers.Conv2D(32, (3, 3), padding='same')(audio_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    audio_features = layers.GlobalAveragePooling2D()(x)
    audio_features = layers.Dense(
        audio_feature_dim,
        activation='relu',
        name='audio_projection'
    )(audio_features)

    # ============ Fusion Head ============
    # Concatenate features
    fused = layers.Concatenate(name='fusion_concat')(
        [vision_features, audio_features]
    )

    # Classification head
    x = layers.Dense(256, activation='relu', name='fusion_dense1')(fused)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu', name='fusion_dense2')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output
    output = layers.Dense(
        num_classes,
        activation='softmax',
        name='output'
    )(x)

    # Create model
    model = Model(
        inputs=[vision_input, audio_input],
        outputs=output,
        name='fusion_model'
    )

    return model


def FusionModelWithModalityDropout(
    num_classes: int = 10,
    modality_dropout_rate: float = 0.1,
    **kwargs
):
    """
    Fusion model with modality dropout during training.

    Randomly drops one modality to make model robust.
    """
    # TODO: Implement modality dropout layer
    # This is a custom training technique
    return FusionModel(num_classes=num_classes, **kwargs)


def FVFusionModel(
    num_classes: int = 10,
    fv_dim: int = 576,
    audio_feature_dim: int = 128,
    vision_feature_dim: int = 256,
    dropout_rate: float = 0.3,
):
    """
    Fusion model whose vision input is the FROZEN backbone's feature vector
    instead of the raw image (F6 wire format: the edge runs MobileNetV3 and
    ships its 576-float pooled output — frames never leave the device).

    Identical to FusionModel from the vision_projection onward; all trained
    weights (vision_projection, audio branch, fusion head) transfer 1:1 from
    a trained FusionModel — only the frozen backbone moves to the edge.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available")

    fv_input = layers.Input(shape=(fv_dim,), name='vision_fv_input')
    vision_features = layers.Dense(
        vision_feature_dim, activation='relu',
        name='vision_projection')(fv_input)

    audio_input = layers.Input(shape=(51, 128, 1), name='audio_input')
    x = layers.Conv2D(32, (3, 3), padding='same')(audio_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    audio_features = layers.GlobalAveragePooling2D()(x)
    audio_features = layers.Dense(
        audio_feature_dim, activation='relu',
        name='audio_projection')(audio_features)

    fused = layers.Concatenate(name='fusion_concat')(
        [vision_features, audio_features])
    x = layers.Dense(256, activation='relu', name='fusion_dense1')(fused)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu', name='fusion_dense2')(x)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return Model(inputs=[fv_input, audio_input], outputs=output,
                 name='fv_fusion_model')


def transfer_fusion_weights(image_fusion, fv_fusion) -> int:
    """
    Copy every trainable layer's weights from a trained image-input
    FusionModel into an FVFusionModel (backbone excluded — it lives on the
    edge). Matches by layer name where names exist (projections, head,
    output) and by order for the unnamed audio conv/BN stack. Returns the
    number of layers transferred.
    """
    named = ['vision_projection', 'audio_projection',
             'fusion_dense1', 'fusion_dense2', 'output']
    n = 0
    for name in named:
        fv_fusion.get_layer(name).set_weights(
            image_fusion.get_layer(name).get_weights())
        n += 1
    conv_bn = (layers.Conv2D, layers.BatchNormalization)
    src_stack = [l for l in image_fusion.layers if isinstance(l, conv_bn)]
    dst_stack = [l for l in fv_fusion.layers if isinstance(l, conv_bn)]
    for src, dst in zip(src_stack, dst_stack):
        dst.set_weights(src.get_weights())
        n += 1
    return n


def AudioOnlyModel(num_classes: int = 10, dropout_rate: float = 0.3):
    """
    Audio-only model for comparison experiments.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available")

    audio_input = layers.Input(shape=(51, 128, 1), name='audio_input')

    x = layers.Conv2D(32, (3, 3), padding='same')(audio_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    output = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs=audio_input, outputs=output, name='audio_only_model')


# Standalone test
if __name__ == "__main__":
    if TF_AVAILABLE:
        print("Creating fusion model...")
        model = FusionModel(num_classes=10)
        model.summary()

        # Count parameters
        trainable = sum(
            tf.keras.backend.count_params(w)
            for w in model.trainable_weights
        )
        non_trainable = sum(
            tf.keras.backend.count_params(w)
            for w in model.non_trainable_weights
        )

        print(f"\nTrainable parameters: {trainable:,}")
        print(f"Non-trainable parameters: {non_trainable:,}")
        print(f"Total parameters: {trainable + non_trainable:,}")
