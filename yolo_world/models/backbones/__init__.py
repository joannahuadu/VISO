# Copyright (c) Tencent Inc. All rights reserved.
# YOLO Multi-Modal Backbone (Vision Language)
# Vision: YOLOv8 CSPDarknet
# Language: CLIP Text Encoder (12-layer transformer)
from .mm_backbone import (
    MultiModalYOLOBackbone,
    HuggingVisionBackbone,
    HuggingCLIPLanguageBackbone,
    PseudoLanguageBackbone,
    OpenCLIPLanguageBackbone)

__all__ = [
    'MultiModalYOLOBackbone',
    'HuggingVisionBackbone',
    'HuggingCLIPLanguageBackbone',
    'PseudoLanguageBackbone',
    'OpenCLIPLanguageBackbone'
]
