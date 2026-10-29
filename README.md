# ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification

This repository provides the implementation for our paper "ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification". Our approach introduces an adaptive KV cache mixed-precision quantization method for LLMs.

[arXiv](https://arxiv.org/pdf/2405.14256) | [BibTeX](https://scholar.googleusercontent.com/scholar.bib?q=info:0wKnyPeFiagJ:scholar.google.com/&output=citation&scisdr=ClFh5NBhEOb2yBOtPYQ:AFWwaeYAAAAAZqCrJYRNPlzMV0rmEN80DyEui-Y&scisig=AFWwaeYAAAAAZqCrJciDVFyQVLBEdGtvShl5UxQ&scisf=4&ct=citation&cd=-1&hl=zh-CN)

## Getting Started

Follow the step-by-step tutorial to set up ZipCache.

### Step 1: Setup
Create a virtual environment and install dependencies as specified by requirements.txt. Then install flash_attn and zipcache as follows:
```bash
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install -e .

```

### Step 2: Download Pretrained Models
Download the pretrained LLaMA model from huggingface and modify the MODEL_PATH in zipcache_generation_demo.py.

### Step 3: Inference with ZipCache
```python
python3 zipcache_generation_demo.py

```

## Probe-token ratio ablation

The original LLaMA path uses a fixed probe budget with approximately half of the probes from the recent prompt tail and half sampled uniformly at random (with replacement). You can vary the allocation without changing the total probe budget:

```bash
python zipcache_generation_demo.py --probe-ratio 0.10 --recent-probe-share 0.00 --seed 20261029
python zipcache_generation_demo.py --probe-ratio 0.10 --recent-probe-share 0.25 --seed 20261029
python zipcache_generation_demo.py --probe-ratio 0.10 --recent-probe-share 0.50 --seed 20261029
python zipcache_generation_demo.py --probe-ratio 0.10 --recent-probe-share 0.75 --seed 20261029
python zipcache_generation_demo.py --probe-ratio 0.10 --recent-probe-share 1.00 --seed 20261029
```

`--recent-probe-share 0.00` uses only random probes; `1.00` uses only recent probes. Keep the model, prompts, decoding parameters, and random seeds fixed when comparing runs. The demo still requires a local LLaMA checkpoint and the CUDA setup used by the original script. The CPU unit tests for probe selection can be run with `python -m unittest discover -s tests -v`.

## BibTeX
If you find this work useful for your research, please consider citing:
```
@article{he2024zipcache,
  title={ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification},
  author={He, Yefei and Zhang, Luoming and Wu, Weijia and Liu, Jing and Zhou, Hong and Zhuang, Bohan},
  journal={arXiv preprint arXiv:2405.14256},
  year={2024}
}
```
