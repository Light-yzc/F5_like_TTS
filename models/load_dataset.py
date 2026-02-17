from datasets import load_dataset

ds = load_dataset("amphion/Emilia-Dataset", streaming=True)

for sample in ds["train"]:
    meta = sample["json"]  # 包含 text, speaker, duration 等
    audio_bytes = sample["mp3"]
    if meta["language"] == "zh" and meta["dnsmos"] > 3.0:
        process(sample)