from transformers import PretrainedConfig

class DF_Arena_1B_Config(PretrainedConfig):
    model_type = "antispoofing"
    def __init__(self, num_labels=2, sample_rate=16000, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.sample_rate = sample_rate
        self.out_dim = 1024
