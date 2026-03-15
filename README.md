
# VAE-DiT TTS v1.0 Preview

基于 **Diffusion Transformer (DiT)**、**Flow Matching** 与 **VAE latent 建模** 的多语言 Zero-Shot 文本到语音（TTS）模型。只需提供一小段参考音频，即可快速克隆目标说话人的音色，并生成自然流畅的语音。

当前版本为 **v1.0 Preview**：在早期版本基础上，重点优化了 **Duration Predictor（时长预测器）**，并引入了 **GAN-based latent optimization**，以进一步提升生成语音的节奏稳定性、清晰度与整体自然度。

🤗 **Hugging Face 模型页**：[Regeny/VAE-DiT-TTS](https://huggingface.co/Regeny/VAE-DiT-TTS)

---

## 🔔 v1.0 Preview 更新内容

相比先前版本，**v1.0 Preview** 主要包含以下改进：

- ✅ **优化 Duration Predictor**
  - 改善长句与复杂文本下的时长预测稳定性
  - 减少语速异常、提前截断、漏音等问题
  - 提升多语言场景下的节奏一致性

- ✅ **新增 GAN-based latent optimization**
  - 在 VAE latent 空间中引入额外对抗式优化目标
  - 提升生成结果的细节表现与音质还原
  - 改善高频细节、发音边界与整体听感自然度

- ✅ **整体推理效果增强**
  - 更稳定的长文本合成体验
  - 更自然的停顿、韵律与句间过渡
  - 更好的说话人相似度与可懂度平衡

---

## ⚠️ Preview 阶段说明 / Known Limitations

尽管 **v1.0 Preview** 已对 **Duration Predictor** 做了明显优化，但在**超长句子**、**复杂混合语言文本**或**标点稀少的输入**上，自动时长预测仍可能出现偏差。

如果在合成长句时遇到以下情况：

- 声音偏快
- 局部漏音
- 结尾被提前截断
- 停顿节奏不理想

建议在推理时**手动指定 `duration` 参数**。  
例如如果你预估目标输出约为 10 秒，可传入 `duration=10.0`，以获得更稳定的生成结果。

---

## ✨ 特性

- 🌏 **多语言支持**：中文 (ZH)、日文 (JA)、英文 (EN)
- 🎙️ **Zero-shot 声音克隆**：只需 3-10 秒参考音频
- 🏗️ **约 500M 参数量**：基于 DiT 主干与交叉注意力机制
- ⚡ **30 步快速推理**：使用带 Classifier-Free Guidance 的 Sway Sampling
- 🎵 **48kHz 高保真输出**：基于 Oobleck VAE 的高质量音频解码
- ⏱️ **增强的时长控制能力**：优化后的 Duration Predictor 带来更稳定的节奏建模
- 🎚️ **GAN latent optimization**：进一步改善 latent 质量与最终音频表现

---

## 📊 模型架构

| 组件 | 详情 |
|---|---|
| **主干网络 (Backbone)** | DiT (30 层, 1024 维度, 16 头) |
| **文本编码器** | F5 风格的字符级 Tokenizer + ConvNeXt (14 层) |
| **音频编码器** | AutoencoderOobleck (64 维隐空间, 25fps) |
| **时长预测器** | 改进版 Duration Predictor（ConvNeXt + Transformer + 多尺度池化） |
| **latent 优化** | GAN-based latent optimization / refinement |
| **采样方法** | Flow Matching + Sway Sampling (coef=-1.0) |
| **训练损失** | CTC 对齐损失 + Flow Matching 损失 + 对抗式 latent 优化目标 |

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch torchaudio transformers diffusers phonemizer pykakasi pypinyin rjieba huggingface_hub
apt install espeak-ng  # Linux 系统必需，用于 G2P 音素转换
# brew install espeak-ng  # macOS
````

### 2. 代码调用

```python
import torch
from huggingface_hub import hf_hub_download, snapshot_download

# 1. 下载模型权重与代码、VAE等
repo_id = "Regeny/VAE-DiT-TTS"
ckpt_path = hf_hub_download(
    repo_id=repo_id,
    filename="checkpoint.pt"
)
vocab_path = hf_hub_download(
    repo_id=repo_id,
    filename="char_vocab.json"
)

# 下载 VAE 文件夹
vae_dir = snapshot_download(
    repo_id=repo_id,
    allow_patterns=["vae/*"]
)
vae_local_path = f"{vae_dir}/vae"

# 2. 加载模型
from inference import load_checkpoint
from models.vae import load_vae, vae_encode, vae_decode

device = torch.device("cuda")
dit, text_encoder, dur_pred, flow, cfg, tokenizer = load_checkpoint(
    ckpt_path, device, vocab_path_override=vocab_path
)
vae = load_vae(vae_local_path, device=device, precision="fp16")

# 3. 合成语音
from inference import inference

# 注意：如果遇到长句漏音、节奏不稳定或结尾截断，
# 建议取消下方 duration 参数的注释并手动估算时长
inference(
    dit, text_encoder, dur_pred, flow, cfg,
    prompt_audio_path="ref_audio.mp3",          # 你的 3-10 秒参考音频
    prompt_text="誰であれ、戦う心があるのなら──",
    tts_text="你好，今天天气真好！",          # 你想合成的文字
    prompt_language="JA",
    tts_language="ZH",
    char_tokenizer=tokenizer,
    vae_encode_fn=lambda wav: vae_encode(vae, wav),
    vae_decode_fn=lambda lat: vae_decode(vae, lat),
    output_path="output.wav",
    seed=42,
    # duration=5.0  # 手动指定生成音频的总时长(秒)
)
```

### 3. Jupyter Notebook 交互界面

参考本项目提供的 [`inference_notebook.ipynb`](./inference_notebook.ipynb)，可以在 Jupyter / Colab 中直接运行，支持音频播放与批量生成。

---

## 🎧 音频示例

| # | 语言 | 文本                                                                                                                                                         | 音频                                                                |
| - | -- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 1 | EN | We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. | [▶ 播放](./output_example/audio_EN_278000_9eeb8580269dc60eca01.wav) |
| 2 | ZH | 杀死我的责任，你打算怎么负责呢？纯白的吸血姬这么说着。                                                                                                                                | [▶ 播放](./output_example/audio_ZH_283999_cfe560587928520efd53.wav) |
| 3 | JA | 歌に物語を乗せる、全身で感情を表現する。ステージでの活動は流動的で、とても困難です。でも、誰も傷つかない、皆さんが笑顔で一つになる感覚は、とても素晴らしいものだと思います                                                                      | [▶ 播放](./output_example/output.wav)                               |

---

## 📁 主要文件

```text
checkpoint.pt             # 模型权重 (包含 DiT, TextEncoder, DurationPredictor)
char_vocab.json           # 字符词表
inference.py              # 推理核心脚本
inference_notebook.ipynb  # 交互式推理 Notebook
configs/model_medium.yaml # 模型配置
```

---

## ⚙️ 推理参数说明

| 参数          | 默认值  | 说明                            |
| ----------- | ---- | ----------------------------- |
| `cfg_scale` | 3.0  | 分类器引导强度，越大对提示音/文本约束越强，但可能引入失真 |
| `n_steps`   | 30   | 采样步数                          |
| `duration`  | auto | **长句合成时建议手动指定的总时长（秒）**        |
| `seed`      | None | 随机种子                          |

---

## 📝 训练细节

* **数据**：中 / 日 / 英 多语言语音数据集
* **训练步数**：350K
* **预处理**：48kHz 音频，Oobleck VAE (64-dim, 25fps) 隐空间表示
* **对齐方案**：采用 CTC Loss 强制对齐，缓解早期常见的跳字、漏字问题
* **优化方向**：在 v1.0 Preview 中进一步增强了 Duration Predictor，并加入 GAN-based latent optimization

---

## 🙏 致谢

* [F5-TTS](https://github.com/SWivid/F5-TTS) — 文本编码器结构灵感
* [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0) — Oobleck VAE
* [espeak-ng](https://github.com/espeak-ng/espeak-ng) — 多语言 G2P 引擎

---

## 协议

Apache 2.0


