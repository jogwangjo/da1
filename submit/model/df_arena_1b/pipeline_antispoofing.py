from transformers import Pipeline
import torch
from .feature_extraction_antispoofing import AntispoofingFeatureExtractor
class AntispoofingPipeline(Pipeline):
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
        self.feature_extractor = AntispoofingFeatureExtractor()

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        
        if "sampling_rate" in kwargs:
            preprocess_kwargs["sampling_rate"] = kwargs["sampling_rate"]
        
        return preprocess_kwargs, {}, postprocess_kwargs
    
    def preprocess(self, audio, sampling_rate=16000):
        audio = self.feature_extractor(audio)['input_values']
        inputs = {"input_values": audio}
        
        return inputs
    
    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return outputs
    
    def postprocess(self, model_outputs):
        logits = model_outputs['logits']
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()
        
        return {
            "label": self.model.config.id2label[predicted_class],
            "logits": logits.tolist(),
            "score": confidence,
            "all_scores": {
                self.model.config.id2label[i]: probs[0][i].item()
                for i in range(len(probs[0]))
            }
        }