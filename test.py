from data.dataset import TTSDataset

dataset = TTSDataset(
    data_root="/content/F5_like_TTS/processed_dir/train",
    latent_rate=16,
    min_duration_sec=1.0,
    max_duration_sec=20.0,
    prompt_ratio_min=0.05,
    prompt_ratio_max=0.2,
)
import random
print(dataset[70000])