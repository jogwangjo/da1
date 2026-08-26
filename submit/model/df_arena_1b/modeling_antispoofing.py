import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .configuration_antispoofing import DF_Arena_1B_Config
from .backbone import DF_Arena_1B
from .feature_extraction_antispoofing import AntispoofingFeatureExtractor

class DF_Arena_1B_Antispoofing(PreTrainedModel):
    config_class = DF_Arena_1B_Config

    def __init__(self, config: DF_Arena_1B_Config):
        super().__init__(config)
        self.feature_extractor = AntispoofingFeatureExtractor()
        # your backbone here (CNN/TDNN/Wav2Vec front-end, etc.)
        self.backbone = DF_Arena_1B()
        self.post_init()

    def forward(self, input_values, attention_mask=None):
        # input_values: (batch, time) float32 waveform @ config.sample_rate
        logits = self.backbone(input_values)
        return {"logits": logits}
