# Copyright (c) Tencent Inc. All rights reserved.
# Basic brick modules for PAFPN based on CSPLayers

from .yolo_bricks import (
    CSPLayerWithTwoConv,
    MaxSigmoidAttnBlock,
    MaxSigmoidCSPLayerWithTwoConv,
    ImagePoolingAttentionModule,
    RepConvMaxSigmoidCSPLayerWithTwoConv,
    RepMaxSigmoidCSPLayerWithTwoConv
    )
from .yolo_bricks_sp import MaxSigmoidCSPLayerWithTwoConvSPInfer, KnowledgeAttnBlock, MaxSigmoidAttnBlockSPInfer, DownSampleConvSP, DownSampleConvSPInfer

__all__ = ['CSPLayerWithTwoConv',
           'MaxSigmoidAttnBlock',
           'MaxSigmoidCSPLayerWithTwoConv',
           'RepConvMaxSigmoidCSPLayerWithTwoConv',
           'RepMaxSigmoidCSPLayerWithTwoConv',
           'ImagePoolingAttentionModule',
           'KnowledgeAttnBlock',
           'MaxSigmoidCSPLayerWithTwoConvSPInfer', 
           'MaxSigmoidAttnBlockSPInfer',
           'DownSampleConvSP',
           'DownSampleConvSPInfer']
