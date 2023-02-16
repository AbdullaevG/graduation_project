from dataclasses import dataclass, field


@dataclass
class ModelParams:
    """Structure for data parameters"""
    sample_rate: int = field(default=32000)
    window_size: int = field(default=1024)
    hop_size: int = field(default=50)
    mel_bins: int = field(default=64)
    fmin: int = field(default=8)
    fmax: int = field(default=14000)
    classes_num: int = field(default=1)


@dataclass
class EmbeddingModelParams:
    checkpoint_path: str = field(default="Cnn10_mAP=0.380.pth")
    sample_rate: int = field(default=32000)
    window_size: int = field(default=1024)
    hop_size: int = field(default=50)
    mel_bins: int = field(default=64)
    fmin: int = field(default=8)
    fmax: int = field(default=14000)
    classes_num: int = field(default=1)

