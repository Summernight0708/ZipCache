import argparse

import torch
from transformers import AutoTokenizer

from zipcache import MyLlamaForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the ZipCache LLaMA demo with configurable probe-token ratios."
    )
    parser.add_argument(
        "--probe-ratio",
        type=float,
        default=0.10,
        help="Total fraction of prompt positions used as probe tokens (default: 0.10).",
    )
    parser.add_argument(
        "--recent-probe-share",
        type=float,
        default=0.50,
        help="Fraction of the fixed probe budget assigned to recent tokens, from 0 to 1 (default: 0.50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261029,
        help="Random seed for repeatable probe sampling (default: 20261029).",
    )
    args = parser.parse_args()
    if not 0.0 < args.probe_ratio <= 1.0:
        parser.error("--probe-ratio must be in the interval (0, 1]")
    if not 0.0 <= args.recent_probe_share <= 1.0:
        parser.error("--recent-probe-share must be in the interval [0, 1]")
    return args


args = parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
compress_config = {}
## Key compress config
compress_config = {}
compress_config["compress_mode"] = "mixed_channelwiseQ"
compress_config["quantize_bit_important"] = 4
compress_config["quantize_bit_unimportant"] = 2
compress_config["k_unimportant_ratio"] = 0.4
## Value compress config
compress_config["v_compress_mode"] = "channel_separate_mixed_tokenwiseQ"
compress_config["v_quantize_bit_important"] = 4
compress_config["v_quantize_bit_unimportant"] = 2
compress_config["v_unimportant_ratio"] = 0.4
compress_config["stream"] = True # streaming-gear set to true to perform better efficiency
compress_config["streaming_gap"] = 100 # re-compress every N iteration
compress_config["probe_ratio"] = args.probe_ratio
compress_config["recent_probe_share"] = args.recent_probe_share
print(
    f"Probe ratio={args.probe_ratio:.2f}; "
    f"recent probe share={args.recent_probe_share:.0%}; "
    f"random probe share={1.0 - args.recent_probe_share:.0%}; "
    f"seed={args.seed}"
)

MODEL_PATH='/data/models--meta-llama--Meta-Llama-3-8B/snapshots/1460c22666392e470910ce3d44ffeb2ab7dbd4df/' ## your llama path here

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, use_fast=True, cache_dir=MODEL_PATH, local_files_only=True
)
with open('asset/gsm8k_sample.txt', 'r') as file:
    prompt_text = file.read()
input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.cuda()

if 'Llama' in MODEL_PATH:
    model = MyLlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        cache_dir=MODEL_PATH,
        compress_config=compress_config,
        torch_dtype=torch.float16,
        local_files_only=True
    )
else:
    raise NotImplementedError

model.half().eval().cuda()

generate_kwargs = dict(
    return_dict_in_generate=False,
    max_new_tokens=128,
    output_scores=False,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=True,
)

generate_kwargs["do_sample"] = False
generate_kwargs["temperature"] = None
generate_kwargs["top_k"] = None
generate_kwargs["top_p"] = None

generate_ids = model.generate(input_ids, **generate_kwargs)
result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("################## Generated Context with Our Cache ###################")
print(result)
