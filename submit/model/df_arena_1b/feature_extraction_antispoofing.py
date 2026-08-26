from transformers import SequenceFeatureExtractor
import numpy as np
import torch

class AntispoofingFeatureExtractor(SequenceFeatureExtractor):
    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        return_attention_mask=True,
        **kwargs
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs
        )
        self.return_attention_mask = return_attention_mask
    
    def __call__(self, audio, sampling_rate=None, return_tensors=True, **kwargs):
        audio = self.pad(audio, 64600)
        audio = torch.Tensor(audio)
        return {
            "input_values": audio
            
        }

    def pad(self, x, max_len):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        num_repeats = int(max_len / x_len)+1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x	    