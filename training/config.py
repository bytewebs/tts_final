class DeepVoiceConfig:
    def __init__(self):
        self.sample_rate = 22050
        self.hop_size = 256
        self.n_mels = 80
        self.fmin = 0
        self.fmax = 8000
        self.rescaling = False
        self.rescaling_max = 0.999
        self.allow_clipping_in_normalization = False
        self.builder = "deepvoice3_multispeaker"
        
    def get(self, key, default=None):
        """Add get method to make it dict-like"""
        return getattr(self, key, default)
        
    @classmethod
    def from_json(cls, json_path: str) -> 'DeepVoiceConfig':
        """Load config from json file"""
        import json
        with open(json_path) as f:
            config_dict = json.load(f)
        config = cls()
        for k, v in config_dict.items():
            setattr(config, k, v)
        return config 