# Efficient Reasoning with Balanced Thinking

This branch contains the ReBalance implementation for the NPU-native model [openPangu](https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-7B-V1.1).

## 📚 TABLE OF CONTENTS

1. [Environment](#-environment)
2. [Quick Start](#-quick-start)
3. [Reproducing Results](#-reproducing-results)
4. [Acknowledgements](#️-acknowledgements)
5. [Citation](#-citation)

## 📦 Environment

First, download the model from the [official site](https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-7B-V1.1). Then create a conda environment and install dependencies.

```bash
conda create -n pangu python=3.10  
conda activate pangu
pip install -r requirements.txt
```

## 🚀 Quick Start

**1. Steering Vector Extraction**

```bash
# Extract the inference data needed for steering vector generation
bash ./scripts/extract_hidden_pangu_dist.sh

# Extract the steering vector
bash ./scripts/hidden_analysis_pangu.sh 
```

**2. Inference with Dynamic Steering**

```bash
bash ./scripts/dynamic_steer_pangu_dist_conf_all.sh
```

## 📊 Reproducing Results

Run the following script to reproduce our results.

```bash
bash ./scripts/run_all.sh
```

We use the following hyperparameters in our experiments:

| Parameter  | Value  | Parameter  | Value  |
| :---: | :---: | :---: | :---: |
| `steer_layer` | 18 | `seed` | 42 |
| `q25` | 0.75 | `q75` | 0.92 |
| `low_val` | -3.3 | `tau` | 0.01 |
| `max_generated_tokens` | 32000 |

## ❤️ Acknowledgements

Our work builds upon the codebase of [SEAL](https://github.com/VITA-Group/SEAL), [DeepSeek-R1-Distill-Qwen](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), [Qwen3](https://github.com/QwenLM/Qwen3), [QwQ](https://github.com/QwenLM/QwQ), and [openPangu](https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-7B-V1.1). We sincerely thank the authors for their remarkable contributions.

## 🙏 Citation

If you find ReBalance useful in your research, please cite our paper:

```bibtex
@article{li2026efficient,
  title={Efficient Reasoning with Balanced Thinking},
  author={Li, Yulin and Tu, Tengyao and Ding, Li and Wang, Junjie and Zhen, Huiling and Chen, Yixin and Li Yong and Tian, Zhuotao},
  booktitle={Proceedings of the 14th International Conference on Learning Representations},
  year={2026}
}
```
