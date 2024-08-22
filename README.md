# SqueezeLLM: Dense-and-Sparse Quantization [[Paper](https://arxiv.org/abs/2306.07629)]

![Thumbnail](figs/thumbnail.png)


SqueezeLLM is a post-training quantization framework that incorporates a new method called Dense-and-Sparse Quantization to enable efficient LLM serving.

TLDR:
Deploying LLMs is difficult due to their large memory size. This can be addressed with reduced precision quantization. But a naive method hurts performance. We address this with a new Dense-and-Sparse Quantization method.
Dense-and-Sparse splits weight matrices into two components: A dense component that can be heavily quantized without affecting model performance, as well as a sparse part that preserves sensitive and outlier parts of the weight matrices
With this approach, we are able to serve larger models with smaller memory footprint, the same latency, and **yet higher accuracy and quality**.
For instance, the Squeeze variant of the Vicuna models can be served within 6 GB of memory and reach 2% higher MMLU than the baseline model in FP16 with an even 2x larger memory footprint.
For more details please check out our [paper](https://arxiv.org/abs/2306.07629v2).

**Updates (2/5):** Dense and sparse quantization and packing codes for custom models are now available.

**Updates (11/28):** Mistral model is now supported.

**News (10/21):** [SqueezeLLM](https://github.com/vllm-project/vllm/blob/1f24755bf802a2061bd46f3dd1191b7898f13f45/vllm/model_executor/quantization_utils/squeezellm.py#L8) is now supported within the official [vLLM](https://github.com/vllm-project/vllm) framework.

**Updates (9/30):** The code for quantizing custom models is now available ([link](https://github.com/SqueezeAILab/SqueezeLLM#from-scratch-quantization)).

---
## Installation

1. Create a conda environment
```
conda create --name sqllm python=3.9 -y
conda activate sqllm
```

2. Clone and install the dependencies
```
git clone https://github.com/SqueezeAILab/SqueezeLLM
cd SqueezeLLM
pip install -e .
cd squeezellm
python setup_cuda.py install
```

---

## From-scratch Quantization 

To quantize your own models, follow the procedure in this [link](https://github.com/SqueezeAILab/SqueezeLLM/tree/main/quantization). 


## Supported Models

Currently, we support [LLaMA](https://arxiv.org/abs/2302.13971) 7B, 13B, 30B and 65B, [LLaMA-2](https://arxiv.org/abs/2307.09288) 7B and 13B, instruction-tuned [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) 7B and 13B, [XGen](https://blog.salesforceairesearch.com/xgen/) 7B with 8k sequence length, and OPT 1.3B to 30B.
For each model, we support 3-bit and 4-bit quantized models, with sparse levels of 0% (dense-only), 0.05%, and 0.45%.
See our [Paper](https://arxiv.org/abs/2306.07629) for more detailed information on these configurations.
Below are the links to download the models.

### LLaMA (v1)

| Model |  Bitwidth | Dense-only (0%) | 0.05% Sparsity | 0.45% sparsity |
| -------- | -------- | -------- | ------ | ---- |
| LLaMA-7B    | 3   |  [sq-llama-7b-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-llama-7b-w3-s0/blob/main/sq-llama-7b-w3-s0.pt) | [sq-llama-7b-w3-s5](https://huggingface.co/squeeze-ai-lab/sq-llama-7b-w3-s5/blob/main/sq-llama-7b-w3-s5.pt) | [sq-llama-7b-w3-s45](https://huggingface.co/squeeze-ai-lab/sq-llama-7b-w3-s45/blob/main/sq-llama-7b-w3-s45.pt) | 
| LLaMA-7B    | 4   | [sq-llama-7b-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-llama-7b-w4-s0/blob/main/sq-llama-7b-w4-s0.pt) | [sq-llama-7b-w4-s5](https://huggingface.co/squeeze-ai-lab/sq-llama-7b-w4-s5/blob/main/sq-llama-7b-w4-s5.pt) | [sq-llama-7b-w4-s45](https://huggingface.co/squeeze-ai-lab/sq-llama-7b-w4-s45/blob/main/sq-llama-7b-w4-s45.pt) |
| LLaMA-13B    | 3   |  [sq-llama-13b-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-llama-13b-w3-s0/blob/main/sq-llama-13b-w3-s0.pt) | [sq-llama-13b-w3-s5](https://huggingface.co/squeeze-ai-lab/sq-llama-13b-w3-s5/blob/main/sq-llama-13b-w3-s5.pt) | [sq-llama-13b-w3-s45](https://huggingface.co/squeeze-ai-lab/sq-llama-13b-w3-s45/blob/main/sq-llama-13b-w3-s45.pt) | 
| LLaMA-13B    | 4   | [sq-llama-13b-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-llama-13b-w4-s0/blob/main/sq-llama-13b-w4-s0.pt) | [sq-llama-13b-w4-s5](https://huggingface.co/squeeze-ai-lab/sq-llama-13b-w4-s5/blob/main/sq-llama-13b-w4-s5.pt) | [sq-llama-13b-w4-s45](https://huggingface.co/squeeze-ai-lab/sq-llama-13b-w4-s45/blob/main/sq-llama-13b-w4-s45.pt) |
| LLaMA-30B    | 3   |  [sq-llama-30b-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-llama-30b-w3-s0/blob/main/sq-llama-30b-w3-s0.pt) |  [sq-llama-30b-w3-s5](https://huggingface.co/squeeze-ai-lab/sq-llama-30b-w3-s5/blob/main/sq-llama-30b-w3-s5.pt) | [sq-llama-30b-w3-s45](https://huggingface.co/squeeze-ai-lab/sq-llama-30b-w3-s45/blob/main/sq-llama-30b-w3-s45.pt)  |
| LLaMA-30B    | 4   | [sq-llama-30b-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-llama-30b-w4-s0/blob/main/sq-llama-30b-w4-s0.pt) |  [sq-llama-30b-w4-s5](https://huggingface.co/squeeze-ai-lab/sq-llama-30b-w4-s5/blob/main/sq-llama-30b-w4-s5.pt) | [sq-llama-30b-w4-s45](https://huggingface.co/squeeze-ai-lab/sq-llama-30b-w4-s45/blob/main/sq-llama-30b-w4-s45.pt)  |
| LLaMA-65B    | 3   |  [sq-llama-65b-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-llama-65b-w3-s0/blob/main/sq-llama-65b-w3-s0.pt) | [sq-llama-65b-w3-s5](https://huggingface.co/squeeze-ai-lab/sq-llama-65b-w3-s5/blob/main/sq-llama-65b-w3-s5.pt) | [sq-llama-65b-w3-s45](https://huggingface.co/squeeze-ai-lab/sq-llama-65b-w3-s45/blob/main/sq-llama-65b-w3-s45.pt) | 
| LLaMA-65B    | 4   |  [sq-llama-65b-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-llama-65b-w4-s0/blob/main/sq-llama-65b-w4-s0.pt) | [sq-llama-65b-w4-s5](https://huggingface.co/squeeze-ai-lab/sq-llama-65b-w4-s5/blob/main/sq-llama-65b-w4-s5.pt) | [sq-llama-65b-w4-s45](https://huggingface.co/squeeze-ai-lab/sq-llama-65b-w4-s45/blob/main/sq-llama-65b-w4-s45.pt) | 

### LLaMA-2

| Model |  Bitwidth | Dense-only (0%) |
| -------- | -------- | -------- |
| LLaMA-2-7B    | 3   |  [sq-llama-7b-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-llama-2-7b-w3-s0/blob/main/sq-llama-2-7b-w3-s0.pt) | 
| LLaMA-2-7B    | 4   |  [sq-llama-7b-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-llama-2-7b-w4-s0/blob/main/sq-llama-2-7b-w4-s0.pt) | 
| LLaMA-2-13B    | 3   |  [sq-llama-13b-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-llama-2-13b-w3-s0/blob/main/sq-llama-2-13b-w3-s0.pt) | 
| LLaMA-2-13B    | 4   |  [sq-llama-13b-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-llama-2-13b-w4-s0/blob/main/sq-llama-2-13b-w4-s0.pt) | 

### Mistral

| Model |  Bitwidth | Dense-only (0%) |
| -------- | -------- | -------- |
| Mistral-7B    | 3   |  [sq-mistral-7b-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-mistral-7b-w3-s0/blob/main/sq-mistral-7b-w3-s0.pt) | 
| Mistral-7B    | 4   |  [sq-mistral-7b-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-mistral-7b-w4-s0/blob/main/sq-mistral-7b-w4-s0.pt) | 
| Mistral-7B-instruct    | 3  |  [sq-mistral-7b-instruct-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-mistral-7b-instruct-w3-s0/blob/main/sq-mistral-7b-instruct-w3-s0.pt) | 
| Mistral-7B-instruct    | 4  |  [sq-mistral-7b-instruct-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-mistral-7b-instruct-w4-s0/blob/main/sq-mistral-7b-instruct-w4-s0.pt) | 

### Vicuna (v1.1)

| Model |  Bitwidth | Dense-only (0%) | 0.45% sparsity |
| -------- | -------- | -------- | ---- |
| Vicuna-7B    | 3   | [sq-vicuna-7b-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-vicuna-7b-w3-s0/blob/main/sq-vicuna-7b-w3-s0.pt) | [sq-vicuna-7b-w3-s45](https://huggingface.co/squeeze-ai-lab/sq-vicuna-7b-w3-s45/blob/main/sq-vicuna-7b-w3-s45.pt)  |
| Vicuna-7B    | 4     | [sq-vicuna-7b-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-vicuna-7b-w4-s0/blob/main/sq-vicuna-7b-w4-s0.pt)  | [sq-vicuna-7b-w4-s45](https://huggingface.co/squeeze-ai-lab/sq-vicuna-7b-w4-s45/blob/main/sq-vicuna-7b-w4-s45.pt) |
| Vicuna-13B    | 3     | [sq-vicuna-13b-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-vicuna-13b-w3-s0/blob/main/sq-vicuna-13b-w3-s0.pt)  | [sq-vicuna-13b-w3-s45](https://huggingface.co/squeeze-ai-lab/sq-vicuna-13b-w3-s45/blob/main/sq-vicuna-13b-w3-s45.pt) |
| Vicuna-13B    | 4    | [sq-vicuna-13b-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-vicuna-13b-w4-s0/blob/main/sq-vicuna-13b-w4-s0.pt)  | [sq-vicuna-13b-w4-s45](https://huggingface.co/squeeze-ai-lab/sq-vicuna-13b-w4-s45/blob/main/sq-vicuna-13b-w4-s45.pt) |


### Vicuna (v1.3)

Please refer to the [Fastchat documentation](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md) for more details about the differences between v1.1 vs v1.3.

| Model |  Bitwidth | Dense-only (0%) |
| -------- | -------- | -------- | 
| Vicuna-7B-v1.3    | 3   | [sq-vicuna-7b-v1.3-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-vicuna-7b-v1.3-w3-s0/blob/main/sq-vicuna-7b-v1.3-w3-s0.pt) | 
| Vicuna-7B-v1.3    | 4   | [sq-vicuna-7b-v1.3-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-vicuna-7b-v1.3-w4-s0/blob/main/sq-vicuna-7b-v1.3-w4-s0.pt) | 
| Vicuna-13B-v1.3    | 3   | [sq-vicuna-7b-v1.3-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-vicuna-13b-v1.3-w3-s0/blob/main/sq-vicuna-13b-v1.3-w3-s0.pt) | 
| Vicuna-13B-v1.3    | 4   | [sq-vicuna-7b-v1.3-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-vicuna-13b-v1.3-w4-s0/blob/main/sq-vicuna-13b-v1.3-w4-s0.pt) | 
| Vicuna-30B-v1.3    | 3   | Coming Soon | 
| Vicuna-30B-v1.3    | 4   | Coming Soon | 

### XGen (8k Sequence length)
[XGen-7B-8k-Base](https://huggingface.co/Salesforce/xgen-7b-8k-base) is a 7B model pre-trained under 8K sequence length.
[XGen-7B-8k-Inst](https://huggingface.co/Salesforce/xgen-7b-8k-inst) is a supervised finetuned model on public domain instructional data for instruction following applications.
Please refer to the [blog post](https://blog.salesforceairesearch.com/xgen/) from Salesforce AI Research for more details on the models.

| Model |  Bitwidth | Dense-only (0%) | 0.45% sparsity |
| -------- | -------- | -------- | ---- |
| XGen-7B-8k-Base    | 3   | [sq-xgen-7b-8k-base-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-xgen-7b-8k-base-w3-s0/blob/main/sq-xgen-7b-8k-base-w3-s0.pt)  | [sq-xgen-7b-8k-base-w3-s45](https://huggingface.co/squeeze-ai-lab/sq-xgen-7b-8k-base-w3-s45/blob/main/sq-xgen-7b-8k-base-w3-s45.pt) |
| XGen-7B-8k-Base    | 4     | [sq-xgen-7b-8k-base-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-xgen-7b-8k-base-w4-s0/blob/main/sq-xgen-7b-8k-base-w4-s0.pt)  | [sq-xgen-7b-8k-base-w4-s45](https://huggingface.co/squeeze-ai-lab/sq-xgen-7b-8k-base-w4-s45/blob/main/sq-xgen-7b-8k-base-w4-s45.pt) |
| XGen-7B-8k-Inst    | 3     | [sq-xgen-7b-8k-inst-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-xgen-7b-8k-inst-w3-s0/blob/main/sq-xgen-7b-8k-inst-w3-s0.pt)  | [sq-xgen-7b-8k-inst-w3-s45](https://huggingface.co/squeeze-ai-lab/sq-xgen-7b-8k-inst-w3-s45/blob/main/sq-xgen-7b-8k-inst-w3-s45.pt) |
| XGen-7B-8k-Inst    | 4     | [sq-xgen-7b-8k-inst-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-xgen-7b-8k-inst-w4-s0/blob/main/sq-xgen-7b-8k-inst-w4-s0.pt)  | [sq-xgen-7b-8k-inst-w4-s45](https://huggingface.co/squeeze-ai-lab/sq-xgen-7b-8k-inst-w4-s45/blob/main/sq-xgen-7b-8k-inst-w4-s45.pt) |

### OPT 

| Model |  Bitwidth | Dense-only (0%) | 0.45% sparsity |
| -------- | -------- | -------- | ---- |
| OPT-1.3B   | 3   | [sq-opt-1.3b-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-opt-1.3b-w3-s0/blob/main/sq-opt-1.3b-w3-s0.pt)  | [sq-opt-1.3b-w3-s50](https://huggingface.co/squeeze-ai-lab/sq-opt-1.3b-w3-s50/blob/main/sq-opt-1.3b-w3-s50.pt) |
| OPT-1.3B   | 4   | [sq-opt-1.3b-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-opt-1.3b-w4-s0/blob/main/sq-opt-1.3b-w4-s0.pt)  | [sq-opt-1.3b-w4-s50](https://huggingface.co/squeeze-ai-lab/sq-opt-1.3b-w4-s50/blob/main/sq-opt-1.3b-w4-s50.pt)  |
| OPT-2.7B   | 3   | [sq-opt-2.7b-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-opt-2.7b-w3-s0/blob/main/sq-opt-2.7b-w3-s0.pt)  | [sq-opt-2.7b-w3-s50](https://huggingface.co/squeeze-ai-lab/sq-opt-2.7b-w3-s50/blob/main/sq-opt-2.7b-w3-s50.pt) |
| OPT-2.7B   | 4   | [sq-opt-2.7b-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-opt-2.7b-w4-s0/blob/main/sq-opt-2.7b-w4-s0.pt)  | [sq-opt-2.7b-w4-s50](https://huggingface.co/squeeze-ai-lab/sq-opt-2.7b-w4-s50/blob/main/sq-opt-2.7b-w4-s50.pt) |
| OPT-6.7B   | 3   | [sq-opt-6.7b-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-opt-6.7b-w3-s0/blob/main/sq-opt-6.7b-w3-s0.pt)  | [sq-opt-6.7b-w3-s50](https://huggingface.co/squeeze-ai-lab/sq-opt-6.7b-w3-s50/blob/main/sq-opt-6.7b-w3-s50.pt) |
| OPT-6.7B   | 4   | [sq-opt-6.7b-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-opt-6.7b-w4-s0/blob/main/sq-opt-6.7b-w4-s0.pt)  | [sq-opt-6.7b-w4-s50](https://huggingface.co/squeeze-ai-lab/sq-opt-6.7b-w4-s50/blob/main/sq-opt-6.7b-w4-s50.pt) |
| OPT-13B   | 3   | [sq-opt-13b-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-opt-13b-w3-s0/blob/main/sq-opt-13b-w3-s0.pt)  | [sq-opt-13b-w3-s50](https://huggingface.co/squeeze-ai-lab/sq-opt-13b-w3-s50/blob/main/sq-opt-13b-w3-s50.pt) |
| OPT-13B   | 4   | [sq-opt-13b-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-opt-13b-w4-s0/blob/main/sq-opt-13b-w4-s0.pt)  | [sq-opt-13b-w4-s50](https://huggingface.co/squeeze-ai-lab/sq-opt-13b-w4-s50/blob/main/sq-opt-13b-w4-s50.pt) |
| OPT-30B   | 3   | [sq-opt-30b-w3-s0](https://huggingface.co/squeeze-ai-lab/sq-opt-30b-w3-s0/blob/main/sq-opt-30b-w3-s0.pt)  | [sq-opt-30b-w3-s50](https://huggingface.co/squeeze-ai-lab/sq-opt-30b-w3-s50/blob/main/sq-opt-30b-w3-s50.pt) |
| OPT-30B   | 4   | [sq-opt-30b-w4-s0](https://huggingface.co/squeeze-ai-lab/sq-opt-30b-w4-s0/blob/main/sq-opt-30b-w4-s0.pt)  | [sq-opt-30b-w4-s50](https://huggingface.co/squeeze-ai-lab/sq-opt-30b-w4-s50/blob/main/sq-opt-30b-w4-s50.pt) |
---

## Running the Models

### Benchmarking

The following code will run and benchmark the 3-bit quantized models on the C4 dataset. 
The `--torch_profile` argument can be passed when running benchmarking to replicate the runtime results from the paper.
Download the quantized model (e.g. `sq-llama-7b-w3-s0.pt` or `sq-xgen-7b-8k-base-w3-s0.py`) locally from the links above.

Note that for the LLaMA (v1) and Vicuna v1.1 models, you need to first obtain the original, pre-trained LLaMA model in the Huggingface-compatible format locally and provide the path in `{model_path}`.
For other model types (e.g. Vicuna v1.3, LLaMA-2, XGen, etc.), you don't need to install/download the original models separately as we provide Huggingface compatible configs of all supported models in `models`. 
You can follow the same procedure for other model types and quantization settings such as bit width and sparsity level.

```
# LLaMA Benchmarking
CUDA_VISIBLE_DEVICES=0 python llama.py {model_path} c4 --wbits 3 --load sq-llama-7b-w3-s0.pt --benchmark 128 --check --torch_profile

# XGen Benchmarking
CUDA_VISIBLE_DEVICES=0 python llama.py models/xgen-7b-8k-base c4 --wbits 3 --load sq-xgen-7b-8k-base-w3-s0.pt --benchmark 128 --check --torch_profile
```

When using checkpoints with sparsity (i.e. non-zero sparsity level), the `--include_sparse` flag should also be passed:
```
# LLaMA Benchmarking
CUDA_VISIBLE_DEVICES=0 python llama.py {model_path} c4 --wbits 3 --load sq-llama-7b-w3-s5.pt --include_sparse --benchmark 128 --check --torch_profile

# XGen Benchmarking
CUDA_VISIBLE_DEVICES=0 python llama.py models/xgen-7b-8k-base c4 --wbits 3 --load sq-xgen-7b-8k-base-w3-s0.pt --include_sparse --benchmark 128 --check --torch_profile
```

**NOTE:** In order to reproduce the perplexity numbers in our paper, please use `--eval` instead of `--benchmark`, following the instruction below.

### Perplexity Evaluation

The following code will evaluate perplexity using the 3-bit quantized models on the C4 dataset, 
following the same evaluation methodology of [GPTQ](https://github.com/IST-DASLab/gptq) and [GPTQ-For-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa/).
This will reproduce the perplexity numbers reported in our paper.
Download the quantized model (e.g. `sq-llama-7b-w3-s0.pt` or `sq-xgen-7b-8k-base-w3-s0.py`) locally from the links above.


Note that for the LLaMA (v1) and Vicuna v1.1 models, you need to first obtain the original, pre-trained LLaMA model in the Huggingface-compatible format locally and provide the path in `{model_path}`.
For other model types (e.g. Vicuna v1.3, LLaMA-2, XGen, etc.), you don't need to install/download the original models separately as we provide Huggingface compatible configs of all supported models in `models`. 
You can follow the same procedure for other model types and quantization settings such as bit width and sparsity level.

```
# LLaMA Perplexity Evaluation
CUDA_VISIBLE_DEVICES=0 python llama.py {model_path} c4 --wbits 3 --load sq-llama-7b-w3-s0.pt --eval

# XGen Perplexity Evaluation
CUDA_VISIBLE_DEVICES=0 python llama.py models/xgen-7b-8k-base c4 --wbits 3 --load sq-xgen-7b-8k-base-w3-s0.pt --eval
```

When using checkpoints with sparsity (i.e. non-zero sparsity level), the `--include_sparse` flag should also be passed:
```
# LLaMA Perplexity Evaluation
CUDA_VISIBLE_DEVICES=0 python llama.py {model_path} c4 --wbits 3 --load sq-llama-7b-w3-s0.pt --include_sparse --eval

# XGen Perplexity Evaluation
CUDA_VISIBLE_DEVICES=0 python llama.py models/xgen-7b-8k-base c4 --wbits 3 --load sq-xgen-7b-8k-base-w3-s0.pt --include_sparse --eval
```

The code was tested on A5000 and A6000 GPUs with Cuda 11.3 and CUDNN 8.2.

---
## Acknowledgement

This code reuses components from several libraries including [GPTQ](https://github.com/IST-DASLab/gptq) as well as [GPTQ-For-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa/).


---

## Citation

SqueezeLLM has been developed as part of the following paper. We appreciate it if you would please cite the following paper if you found the library useful for your work:

```
@article{kim2023squeezellm,
  title={SqueezeLLM: Dense-and-Sparse Quantization},
  author={Kim, Sehoon and Hooper, Coleman and Gholami, Amir and Dong, Zhen and Li, Xiuyu and Shen, Sheng and Mahoney, Michael and Keutzer, Kurt},
  journal={arXiv},
  year={2023}
}
```

=================================================================================================================
My conda environment: 
(sqllm) ayu1@ortce-a100-80G2:~/SqueezeLLM/SqueezeLLM$ conda info --envs
# conda environments:
#
sqllm                 *  /nfs/site/home/ayu1/.conda/envs/sqllm
usr_intelpython          /nfs/site/home/ayu1/.conda/envs/usr_intelpython
                         /nfs/site/home/ayu1/.julia/scratchspaces/8f75cd03-7ff8-4ecb-9b8f-daf728133b1b/conda
                         /nfs/site/home/ayu1/intel/ipex/intelpython/latest
                         /nfs/site/home/ayu1/intel/ipex/intelpython/latest/envs/2023.2.0
                         /nfs/site/home/ayu1/intel/ipex/intelpython/latest/envs/pytorch
                         /nfs/site/home/ayu1/intel/ipex/intelpython/latest/envs/pytorch-gpu
                         /nfs/site/home/ayu1/intel/ipex/intelpython/latest/envs/tensorflow
                         /nfs/site/home/ayu1/intel/ipex/intelpython/latest/envs/tensorflow-gpu
                         /nfs/site/home/ayu1/intel/ipex/pytorch/2.0.1.0
                         /nfs/site/home/ayu1/intel/ipex/tensorflow/2.13.0
base                     /nfs/site/home/ayu1/miniconda3




(sqllm) ayu1@ortce-a100-80G2:~/SqueezeLLM/SqueezeLLM$ pip list -v
Package                     Version                Editable project location                                                                                 Location                                                                                                  Installer
--------------------------- ---------------------- --------------------------------------------------------------------------------------------------------- --------------------------------------------------------------------------------------------------------- ---------
accelerate                  0.31.0                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
aiohttp                     3.9.5                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
aiosignal                   1.3.1                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
annotated-types             0.7.0                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
async-timeout               4.0.3                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
attrs                       23.2.0                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
certifi                     2024.6.2                                                                                                                         /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
charset-normalizer          3.3.2                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
datasets                    2.19.2                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
dill                        0.3.8                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
et-xmlfile                  1.1.0                                                                                                                            /nfs/site/home/ayu1/.local/lib/python3.9/site-packages                                                    pip
filelock                    3.14.0                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
frozenlist                  1.4.1                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
fsspec                      2024.3.1                                                                                                                         /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
huggingface-hub             0.23.3                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
idna                        3.7                                                                                                                              /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
intel-extension-for-pytorch 2.1.20+xpu                                                                                                                       /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
Jinja2                      3.1.4                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
MarkupSafe                  2.1.5                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
mpmath                      1.3.0                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
multidict                   6.0.5                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
multiprocess                0.70.16                                                                                                                          /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
networkx                    3.2.1                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
ninja                       1.10.2.3                                                                                                                         /nfs/site/home/ayu1/.local/lib/python3.9/site-packages                                                    pip
numpy                       1.26.4                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
nvidia-cublas-cu12          12.1.3.1                                                                                                                         /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
nvidia-cuda-cupti-cu12      12.1.105                                                                                                                         /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
nvidia-cuda-nvrtc-cu12      12.1.105                                                                                                                         /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
nvidia-cuda-runtime-cu12    12.1.105                                                                                                                         /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
nvidia-cudnn-cu12           8.9.2.26                                                                                                                         /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
nvidia-cufft-cu12           11.0.2.54                                                                                                                        /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
nvidia-curand-cu12          10.3.2.106                                                                                                                       /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
nvidia-cusolver-cu12        11.4.5.107                                                                                                                       /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
nvidia-cusparse-cu12        12.1.0.106                                                                                                                       /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
nvidia-nccl-cu12            2.20.5                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
nvidia-nvjitlink-cu12       12.5.40                                                                                                                          /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
nvidia-nvtx-cu12            12.1.105                                                                                                                         /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
oneccl-bind-pt              2.1.200+xpu                                                                                                                      /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
openpyxl                    3.1.2                                                                                                                            /nfs/site/home/ayu1/.local/lib/python3.9/site-packages                                                    pip
packaging                   24.1                                                                                                                             /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
pandas                      2.2.2                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
pillow                      10.4.0                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
pip                         23.1.2                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages
popcorn                     0.0.2                                                                                                                            /nfs/site/home/ayu1/.local/lib/python3.9/site-packages                                                    pip
prettytable                 3.9.0                                                                                                                            /nfs/site/home/ayu1/.local/lib/python3.9/site-packages                                                    pip
psutil                      6.0.0                                                                                                                            /nfs/site/home/ayu1/.local/lib/python3.9/site-packages                                                    pip
pyarrow                     16.1.0                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
pyarrow-hotfix              0.6                                                                                                                              /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
pydantic                    2.8.2                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
pydantic_core               2.20.1                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
python-dateutil             2.9.0.post0                                                                                                                      /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
pytz                        2024.1                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
PyYAML                      6.0.1                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
quant-cuda                  0.0.0                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages/quant_cuda-0.0.0-py3.9-linux-x86_64.egg
quant_sycl                  0.0.0                  /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages/quant_sycl-0.0.0-py3.9-linux-x86_64.egg /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages/quant_sycl-0.0.0-py3.9-linux-x86_64.egg
regex                       2024.5.15                                                                                                                        /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
requests                    2.32.3                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
safetensors                 0.4.3                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
sentencepiece               0.2.0                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
setuptools                  69.5.1                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
six                         1.16.0                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
squeezellm                  0.1.0                  /nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM                                                                 /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
sympy                       1.12.1                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
tiktoken                    0.7.0                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
tokenizers                  0.13.3                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
torch                       2.1.0.post0+cxx11.abi                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
torchaudio                  2.1.0.post0+cxx11.abi                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
torchvision                 0.16.0.post0+cxx11.abi                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
tqdm                        4.66.4                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
transformers                4.29.0                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
triton                      2.3.1                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
typing_extensions           4.12.2                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
tzdata                      2024.1                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
urllib3                     2.2.1                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
wcwidth                     0.2.13                                                                                                                           /nfs/site/home/ayu1/.local/lib/python3.9/site-packages                                                    pip
wheel                       0.40.0                                                                                                                           /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages
xxhash                      3.4.1                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip
yarl                        1.9.4                                                                                                                            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages                                         pip


Make sure to set the network proxy and download sq-xgen-7b-8k-base-w3-s0.pt: 
(sqllm) ayu1@ortce-a100-80G2:~/SqueezeLLM/SqueezeLLM$ wget https://huggingface.co/squeeze-ai-lab/sq-xgen-7b-8k-base-w3-s0/resolve/main/sq-xgen-7b-8k-base-w3-s0.pt
--2024-06-11 13:19:04--  https://huggingface.co/squeeze-ai-lab/sq-xgen-7b-8k-base-w3-s0/resolve/main/sq-xgen-7b-8k-base-w3-s0.pt
Resolving proxy-dmz.intel.com (proxy-dmz.intel.com)... 10.7.211.16
Connecting to proxy-dmz.intel.com (proxy-dmz.intel.com)|10.7.211.16|:912... connected.
Proxy request sent, awaiting response... 302 Found
Location: https://cdn-lfs.huggingface.co/repos/fb/e6/fbe61661edb19b3717b30ce8923fac7e704918a8026af6059782ca51f845eb4e/927df6209022a886f15cdb893bca1387e8ad2db0aea95d0d1a281e59e118d51c?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27sq-xgen-7b-8k-base-w3-s0.pt%3B+filename%3D%22sq-xgen-7b-8k-base-w3-s0.pt%22%3B&Expires=1718396344&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxODM5NjM0NH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy9mYi9lNi9mYmU2MTY2MWVkYjE5YjM3MTdiMzBjZTg5MjNmYWM3ZTcwNDkxOGE4MDI2YWY2MDU5NzgyY2E1MWY4NDVlYjRlLzkyN2RmNjIwOTAyMmE4ODZmMTVjZGI4OTNiY2ExMzg3ZThhZDJkYjBhZWE5NWQwZDFhMjgxZTU5ZTExOGQ1MWM%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qIn1dfQ__&Signature=adribaPvcuyUTZ7ZDSfTpQYaydEYLXMjA66JNSUZaJBMXoQ0zmLAK4wI-fVm8TLHU9OlpCtK3bFv1T1BoLhdNqeNofU29kBgoMS4cMwB%7EbQ%7EKK3nj3Rq%7EIwnRhvPjHJuPHP1P-zcJn6Alj5oe2OheTBn03benreYpH1Qv3NuFoamrqwa-5QxRn0sJVr5-3tfX%7EUi93yOI3ffnU35qUWodDmQ8Rvz6f8hLiwpdf7zhTzkvH6hyWorN6g6as-x3SbJ3CKcTxJYpThdV4aKrJrXC8euHX7FncUkRORD84-THQcmlYuxKTL4%7E6IWFocPLOhH3pCcQaRzeL64MiGEPyVzYw__&Key-Pair-Id=KVTP0A1DKRTAX [following]
--2024-06-11 13:19:04--  https://cdn-lfs.huggingface.co/repos/fb/e6/fbe61661edb19b3717b30ce8923fac7e704918a8026af6059782ca51f845eb4e/927df6209022a886f15cdb893bca1387e8ad2db0aea95d0d1a281e59e118d51c?response-content-disposition=inline%3B+filename*%3DUTF-8''sq-xgen-7b-8k-base-w3-s0.pt%3B+filename%3D%22sq-xgen-7b-8k-base-w3-s0.pt%22%3B&Expires=1718396344&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxODM5NjM0NH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy9mYi9lNi9mYmU2MTY2MWVkYjE5YjM3MTdiMzBjZTg5MjNmYWM3ZTcwNDkxOGE4MDI2YWY2MDU5NzgyY2E1MWY4NDVlYjRlLzkyN2RmNjIwOTAyMmE4ODZmMTVjZGI4OTNiY2ExMzg3ZThhZDJkYjBhZWE5NWQwZDFhMjgxZTU5ZTExOGQ1MWM~cmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qIn1dfQ__&Signature=adribaPvcuyUTZ7ZDSfTpQYaydEYLXMjA66JNSUZaJBMXoQ0zmLAK4wI-fVm8TLHU9OlpCtK3bFv1T1BoLhdNqeNofU29kBgoMS4cMwB~bQ~KK3nj3Rq~IwnRhvPjHJuPHP1P-zcJn6Alj5oe2OheTBn03benreYpH1Qv3NuFoamrqwa-5QxRn0sJVr5-3tfX~Ui93yOI3ffnU35qUWodDmQ8Rvz6f8hLiwpdf7zhTzkvH6hyWorN6g6as-x3SbJ3CKcTxJYpThdV4aKrJrXC8euHX7FncUkRORD84-THQcmlYuxKTL4~6IWFocPLOhH3pCcQaRzeL64MiGEPyVzYw__&Key-Pair-Id=KVTP0A1DKRTAX
Connecting to proxy-dmz.intel.com (proxy-dmz.intel.com)|10.7.211.16|:912... connected.
Proxy request sent, awaiting response... 200 OK
Length: 4151020464 (3.9G) [binary/octet-stream]
Saving to: 'sq-xgen-7b-8k-base-w3-s0.pt'

sq-xgen-7b-8k-base-w3-s0.pt                                 100%[========================================================================================================================================>]   3.87G  32.1MB/s    in 2m 23s

2024-06-11 13:21:31 (27.7 MB/s) - 'sq-xgen-7b-8k-base-w3-s0.pt' saved [4151020464/4151020464]

Run the model: 
(sqllm) ayu1@ortce-a100-80G2:~/SqueezeLLM/SqueezeLLM$ CUDA_VISIBLE_DEVICES=0 python llama.py models/xgen-7b-8k-base c4 --wbits 3 --load sq-xgen-7b-8k-base-w3-s0.pt --benchmark 128 --check --torch_profile
models/xgen-7b-8k-base
Loading model ...
Done.
Using unk_token, but it is not set yet.
Using unk_token, but it is not set yet.
STAGE:2024-06-11 17:32:15 74118:74118 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
Model type : llama
Benchmarking ...
0 0.6279926300048828
1 0.023001909255981445
2 0.02353501319885254
3 0.022405385971069336
4 0.022412538528442383
5 0.023001432418823242
6 0.022295236587524414
7 0.02235698699951172
8 0.02310800552368164
9 0.02260756492614746
10 0.022258996963500977
11 0.02302384376525879
12 0.024077415466308594
13 0.022531986236572266
14 0.023962020874023438
15 0.022347211837768555
16 0.02269148826599121
17 0.024286270141601562
18 0.022745132446289062
19 0.022759675979614258
20 0.024289846420288086
21 0.022903919219970703
22 0.022726774215698242
23 0.024248600006103516
24 0.02269744873046875
25 0.023121118545532227
26 0.024370193481445312
27 0.022704124450683594
28 0.022610902786254883
29 0.024161100387573242
30 0.022658348083496094
31 0.02267765998840332
32 0.025534629821777344
33 0.022733449935913086
34 0.022515058517456055
35 0.02417778968811035
36 0.022693395614624023
37 0.022540569305419922
38 0.025284767150878906
39 0.02267742156982422
40 0.02275252342224121
41 0.024185657501220703
42 0.025180816650390625
43 0.023032188415527344
44 0.0243222713470459
45 0.022665739059448242
46 0.022667646408081055
47 0.024109363555908203
48 0.022665739059448242
49 0.022679805755615234
50 0.02431941032409668
51 0.022641897201538086
52 0.022755861282348633
53 0.024183988571166992
54 0.022837162017822266
55 0.024121522903442383
56 0.022785663604736328
57 0.02272963523864746
58 0.02426457405090332
59 0.02278876304626465
60 0.022572755813598633
61 0.02425837516784668
62 0.022715330123901367
63 0.02343583106994629
64 0.024728775024414062
65 0.02297210693359375
66 0.022734642028808594
67 0.02424788475036621
68 0.022727012634277344
69 0.022710561752319336
70 0.024393081665039062
71 0.02274012565612793
72 0.02273273468017578
73 0.024315834045410156
74 0.022746562957763672
75 0.022657394409179688
76 0.024364233016967773
77 0.022821903228759766
78 0.02270030975341797
79 0.02417778968811035
80 0.022913217544555664
81 0.022710800170898438
82 0.024307966232299805
83 0.022675514221191406
84 0.02269124984741211
85 0.024355649948120117
86 0.022749900817871094
87 0.02275824546813965
88 0.023906946182250977
89 0.0226290225982666
90 0.02245020866394043
91 0.024225711822509766
92 0.022733449935913086
93 0.02273869514465332
94 0.026732683181762695
95 0.022884130477905273
96 0.022739410400390625
97 0.024414777755737305
98 0.02271294593811035
99 0.02279210090637207
100 0.024278879165649414
101 0.022788047790527344
102 0.022674560546875
103 0.024253368377685547
104 0.022754907608032227
105 0.0227811336517334
106 0.024361133575439453
107 0.0227968692779541
108 0.02266240119934082
109 0.024198293685913086
110 0.02272772789001465
111 0.02407240867614746
112 0.022756099700927734
113 0.022827863693237305
114 0.02416849136352539
115 0.022826433181762695
116 0.02274775505065918
117 0.024322032928466797
118 0.022768020629882812
119 0.02263331413269043
120 0.02590346336364746
121 0.0231170654296875
122 0.022816181182861328
123 0.024057626724243164
124 0.023346424102783203
125 0.022756576538085938
126 0.024212121963500977
127 0.022876501083374023
Median: 0.02279043197631836
PPL: 28.74595069885254
max memory(MiB): 4394.431640625
STAGE:2024-06-11 17:32:19 74118:74118 ActivityProfilerController.cpp:320] Completed Stage: Collection
STAGE:2024-06-11 17:32:19 74118:74118 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
VecQuant3MatMulKernelNUQPerChannel(float const*, int...         0.00%       0.000us         0.00%       0.000us       0.000us     506.590ms        52.07%     506.590ms      17.668us         28672
                                              aten::mul         7.54%     189.358ms        11.44%     287.105ms       7.735us      80.133ms         8.24%      87.975ms       2.370us         37120
                                              aten::cat         3.92%      98.277ms         6.51%     163.429ms      10.014us      76.447ms         7.86%      80.004ms       4.902us         16320
                                               aten::mm         0.15%       3.692ms         0.18%       4.441ms      34.695us      63.945ms         6.57%      63.959ms     499.680us           128
std::enable_if<!(false), void>::type internal::gemvx...         0.00%       0.000us         0.00%       0.000us       0.000us      63.945ms         6.57%      63.945ms     499.570us           128
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      55.936ms         5.75%      55.936ms       2.264us         24704
                                              aten::add         5.71%     143.443ms         8.34%     209.430ms       7.272us      48.381ms         4.97%      54.560ms       1.894us         28801
                                       cudaLaunchKernel        41.70%        1.047s        41.70%        1.047s       5.065us      44.787ms         4.60%      44.787ms       0.217us        206656
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us      40.960ms         4.21%      40.960ms       5.000us          8192
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us      35.487ms         3.65%      35.487ms       4.366us          8128
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      33.280ms         3.42%      33.280ms       4.000us          8320
                                             aten::mean         2.48%      62.313ms         4.20%     105.477ms      12.678us      33.276ms         3.42%      35.824ms       4.306us          8320
                                              aten::bmm         3.30%      82.769ms         4.86%     122.023ms      14.895us      29.169ms         3.00%      30.731ms       3.751us          8192
void at::native::index_elementwise_kernel<128, 4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      28.683ms         2.95%      28.683ms       3.501us          8192
                                            aten::index         2.75%      68.926ms         4.82%     121.033ms      14.775us      28.680ms         2.95%      30.599ms       3.735us          8192
                                            aten::fill_         2.85%      71.635ms         6.04%     151.631ms       5.288us      28.674ms         2.95%      36.025ms       1.256us         28673
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      28.674ms         2.95%      28.674ms       1.000us         28673
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      27.480ms         2.82%      27.480ms       1.664us         16510
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      24.197ms         2.49%      24.197ms       1.949us         12416
                                              aten::neg         1.92%      48.212ms         3.58%      89.895ms      10.974us      20.052ms         2.06%      21.533ms       2.629us          8192
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      20.052ms         2.06%      20.052ms       2.448us          8192
std::enable_if<!(false), void>::type internal::gemvx...         0.00%       0.000us         0.00%       0.000us       0.000us      15.441ms         1.59%      15.441ms       3.770us          4096
std::enable_if<!(false), void>::type internal::gemvx...         0.00%       0.000us         0.00%       0.000us       0.000us      13.664ms         1.40%      13.664ms       3.362us          4064
                                            aten::rsqrt         1.74%      43.571ms         3.67%      92.140ms      11.075us      12.332ms         1.27%      14.405ms       1.731us          8320
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      12.332ms         1.27%      12.332ms       1.482us          8320
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      10.564ms         1.09%      10.564ms       1.270us          8321
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      10.487ms         1.08%      10.487ms       2.560us          4096
                                         aten::_softmax         0.90%      22.503ms         2.45%      61.415ms      14.994us       9.456ms         0.97%      10.120ms       2.471us          4096
                                              aten::pow         2.17%      54.388ms         4.52%     113.328ms      13.621us       8.320ms         0.86%      10.901ms       1.310us          8320
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       8.320ms         0.86%       8.320ms       1.000us          8320
                                             aten::silu         0.96%      24.182ms         2.06%      51.742ms      12.632us       8.192ms         0.84%       8.935ms       2.181us          4096
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       8.192ms         0.84%       8.192ms       2.000us          4096
                                          aten::maximum         1.06%      26.599ms         2.67%      66.947ms      16.344us       7.418ms         0.76%       8.120ms       1.982us          4096
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       7.418ms         0.76%       7.418ms       1.811us          4096
                                              aten::div         0.99%      24.818ms         2.08%      52.277ms      12.760us       6.843ms         0.70%       7.491ms       1.828us          4097
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       6.843ms         0.70%       6.843ms       1.670us          4097
void (anonymous namespace)::softmax_warp_forward<flo...         0.00%       0.000us         0.00%       0.000us       0.000us       5.364ms         0.55%       5.364ms       2.619us          2048
                                     aten::_log_softmax         0.05%       1.141ms         0.08%       2.061ms      16.228us       3.148ms         0.32%       3.168ms      24.945us           127
void at::native::(anonymous namespace)::cunn_SoftMax...         0.00%       0.000us         0.00%       0.000us       0.000us       3.148ms         0.32%       3.148ms      24.787us           127
void (anonymous namespace)::softmax_warp_forward<flo...         0.00%       0.000us         0.00%       0.000us       0.000us       2.048ms         0.21%       2.048ms       2.000us          1024
void (anonymous namespace)::softmax_warp_forward<flo...         0.00%       0.000us         0.00%       0.000us       0.000us       1.024ms         0.11%       1.024ms       2.000us           512
void (anonymous namespace)::softmax_warp_forward<flo...         0.00%       0.000us         0.00%       0.000us       0.000us     512.000us         0.05%     512.000us       2.000us           256
                                     aten::index_select         0.42%      10.479ms         9.00%     225.843ms       1.764ms     388.000us         0.04%     406.000us       3.172us           128
void at::native::(anonymous namespace)::indexSelectS...         0.00%       0.000us         0.00%       0.000us       0.000us     388.000us         0.04%     388.000us       3.031us           128
                                            aten::copy_         0.05%       1.212ms         0.78%      19.661ms      76.502us     385.000us         0.04%     436.000us       1.696us           257
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us     256.000us         0.03%     256.000us       2.000us           128
void (anonymous namespace)::softmax_warp_forward<flo...         0.00%       0.000us         0.00%       0.000us       0.000us     256.000us         0.03%     256.000us       2.000us           128
                                 aten::nll_loss_forward         0.06%       1.627ms         0.64%      16.133ms     127.031us     254.000us         0.03%     275.000us       2.165us           127
void at::native::(anonymous namespace)::nll_loss_for...         0.00%       0.000us         0.00%       0.000us       0.000us     254.000us         0.03%     254.000us       2.000us           127
                                     aten::masked_fill_         0.08%       2.048ms         0.10%       2.588ms      20.219us     238.000us         0.02%     257.000us       2.008us           128
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     238.000us         0.02%     238.000us       1.859us           128
                                              aten::sub         0.17%       4.149ms         0.97%      24.388ms     190.531us     201.000us         0.02%     217.000us       1.695us           128
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     201.000us         0.02%     201.000us       1.570us           128
                                             aten::add_         0.13%       3.285ms         0.15%       3.696ms      29.333us     150.000us         0.02%     180.000us       1.429us           126
                                           aten::arange         0.12%       2.955ms         0.96%      24.054ms      93.961us     129.000us         0.01%     281.000us       1.098us           256
void (anonymous namespace)::elementwise_kernel_with_...         0.00%       0.000us         0.00%       0.000us       0.000us     129.000us         0.01%     129.000us       1.008us           128
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     128.000us         0.01%     128.000us       1.000us           128
void (anonymous namespace)::softmax_warp_forward<flo...         0.00%       0.000us         0.00%       0.000us       0.000us     128.000us         0.01%     128.000us       2.000us            64
void (anonymous namespace)::softmax_warp_forward<flo...         0.00%       0.000us         0.00%       0.000us       0.000us      64.000us         0.01%      64.000us       2.000us            32
void gemmk1_kernel<int, float, 256, 5, false, false,...         0.00%       0.000us         0.00%       0.000us       0.000us      64.000us         0.01%      64.000us       2.000us            32
void (anonymous namespace)::softmax_warp_forward<flo...         0.00%       0.000us         0.00%       0.000us       0.000us      60.000us         0.01%      60.000us       1.875us            32
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFla...         0.11%       2.877ms         0.11%       2.877ms      11.151us      46.000us         0.00%      46.000us       0.178us           258
                                  cudaDeviceSynchronize         2.35%      58.865ms         2.35%      58.865ms     449.351us      33.000us         0.00%      33.000us       0.252us           131
                                        cudaMemcpyAsync         0.06%       1.613ms         0.06%       1.613ms      12.408us      31.000us         0.00%      31.000us       0.238us           130
                                             cudaMalloc         0.19%       4.750ms         0.19%       4.750ms     113.095us      16.000us         0.00%      16.000us       0.381us            42
                                  cudaStreamIsCapturing         0.00%      38.000us         0.00%      38.000us       0.974us      14.000us         0.00%      14.000us       0.359us            39
                                 cudaDeviceGetAttribute         0.00%       1.000us         0.00%       1.000us       0.067us       5.000us         0.00%       5.000us       0.333us            15
                              aten::_local_scalar_dense         0.00%      33.000us         0.03%     800.000us     800.000us       3.000us         0.00%       3.000us       3.000us             1
                         Memcpy DtoH (Device -> Pinned)         0.00%       0.000us         0.00%       0.000us       0.000us       3.000us         0.00%       3.000us       3.000us             1
                                               cudaFree         0.39%       9.808ms         0.39%       9.808ms       4.904ms       2.000us         0.00%       2.000us       1.000us             2
                                              aten::exp         0.04%       1.127ms         0.05%       1.216ms       1.216ms       2.000us         0.00%       2.000us       2.000us             1
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.000us         0.00%       2.000us       2.000us             1
                       Memcpy HtoD (Pageable -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us       1.000us         0.00%       1.000us       1.000us             1
                                               aten::to         0.07%       1.634ms         0.87%      21.958ms       0.230us       0.000us         0.00%     265.000us       0.003us         95357
                                         aten::_to_copy         0.07%       1.838ms         0.81%      20.341ms     157.682us       0.000us         0.00%     259.000us       2.008us           129
                                    aten::empty_strided         0.03%     861.000us         0.03%     861.000us       6.674us       0.000us         0.00%       0.000us       0.000us           129
                                  cudaStreamSynchronize         0.00%      16.000us         0.00%      16.000us       8.000us       0.000us         0.00%       0.000us       0.000us             2
                                             aten::ones         0.05%       1.299ms         0.75%      18.706ms      18.706ms       0.000us         0.00%       2.000us       2.000us             1
                                            aten::empty         2.85%      71.558ms         2.85%      71.558ms       2.158us       0.000us         0.00%       0.000us       0.000us         33153
                                            aten::slice         1.13%      28.265ms         1.21%      30.355ms       0.725us       0.000us         0.00%       0.000us       0.000us         41854
                                       aten::as_strided         0.11%       2.870ms         0.11%       2.870ms       0.026us       0.000us         0.00%       0.000us       0.000us        112380
                                          aten::reshape         0.66%      16.632ms         0.71%      17.917ms       0.310us       0.000us         0.00%       0.000us       0.000us         57728
                                             aten::view         0.06%       1.390ms         0.06%       1.390ms       0.021us       0.000us         0.00%       0.000us       0.000us         66208
                                          aten::resize_         0.03%     755.000us         0.03%     755.000us       1.971us       0.000us         0.00%       0.000us       0.000us           383
                                        aten::unsqueeze         0.24%       5.985ms         0.24%       6.000ms       0.700us       0.000us         0.00%       0.000us       0.000us          8576
                                        aten::embedding         0.18%       4.413ms         9.17%     230.263ms       1.799ms       0.000us         0.00%     406.000us       3.172us           128
                                           aten::expand         0.48%      11.951ms         0.48%      11.969ms       0.725us       0.000us         0.00%       0.000us       0.000us         16512
                                             aten::rsub         0.02%     569.000us         0.99%      24.931ms     194.773us       0.000us         0.00%     214.000us       1.672us           128
                                      aten::masked_fill         0.17%       4.233ms         0.35%       8.817ms      68.883us       0.000us         0.00%     397.000us       3.102us           128
                                            aten::clone         0.02%     408.000us         0.14%       3.487ms      27.242us       0.000us         0.00%     154.000us       1.203us           128
                                       aten::empty_like         0.01%     188.000us         0.05%       1.225ms       9.570us       0.000us         0.00%       0.000us       0.000us           128
                                      aten::result_type         0.00%       7.000us         0.00%       7.000us       0.001us       0.000us         0.00%       0.000us       0.000us          8320
                                            aten::zeros         3.17%      79.606ms        11.62%     291.750ms      10.175us       0.000us         0.00%      28.416ms       0.991us         28672
                                            aten::zero_         1.93%      48.460ms         7.28%     182.656ms       6.371us       0.000us         0.00%      36.015ms       1.256us         28672
                                        aten::transpose         0.64%      16.042ms         0.64%      16.077ms       0.780us       0.000us         0.00%       0.000us       0.000us         20608
                                          aten::squeeze         0.34%       8.478ms         0.35%       8.789ms       0.536us       0.000us         0.00%       0.000us       0.000us         16384
                                           aten::matmul         2.21%      55.407ms         8.02%     201.326ms      24.198us       0.000us         0.00%      94.690ms      11.381us          8320
                                   cudaGetSymbolAddress         0.02%     553.000us         0.02%     553.000us     553.000us       0.000us         0.00%       0.000us       0.000us             1
                                     aten::_unsafe_view         0.01%     165.000us         0.01%     165.000us       0.020us       0.000us         0.00%       0.000us       0.000us          8320
                                       aten::lift_fresh         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us          4096
                                          aten::detach_         0.18%       4.635ms         0.18%       4.639ms       1.133us       0.000us         0.00%       0.000us       0.000us          4096
                                                detach_         0.00%       4.000us         0.00%       4.000us       0.001us       0.000us         0.00%       0.000us       0.000us          4096
                                              aten::max         0.34%       8.483ms         2.79%      69.929ms      17.073us       0.000us         0.00%       6.801ms       1.660us          4096
                                          aten::softmax         0.35%       8.858ms         2.75%      69.125ms      16.876us       0.000us         0.00%       9.788ms       2.390us          4096
                                           aten::linear         0.02%     461.000us         0.23%       5.718ms      44.672us       0.000us         0.00%      59.963ms     468.461us           128
                                                aten::t         0.01%     241.000us         0.01%     371.000us       2.898us       0.000us         0.00%       0.000us       0.000us           128
                                           aten::select         0.03%     639.000us         0.03%     650.000us       2.559us       0.000us         0.00%       0.000us       0.000us           254
                               aten::cross_entropy_loss         0.13%       3.159ms         0.90%      22.526ms     177.370us       0.000us         0.00%       3.443ms      27.110us           127
                                      aten::log_softmax         0.01%     362.000us         0.09%       2.325ms      18.307us       0.000us         0.00%       2.991ms      23.551us           127
                                      aten::nll_loss_nd         0.02%     511.000us         0.66%      16.645ms     131.063us       0.000us         0.00%     246.000us       1.937us           127
                                         aten::nll_loss         0.01%     300.000us         0.65%      16.433ms     129.394us       0.000us         0.00%     275.000us       2.165us           127
                                   aten::_reshape_alias         0.02%     580.000us         0.02%     580.000us       0.143us       0.000us         0.00%       0.000us       0.000us          4064
                                             aten::item         0.00%      11.000us         0.03%     811.000us     811.000us       0.000us         0.00%       3.000us       3.000us             1
                                          cudaHostAlloc         0.03%     744.000us         0.03%     744.000us     744.000us       0.000us         0.00%       0.000us       0.000us             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 2.510s
Self CUDA time total: 972.813ms





1.pip install tiktoken, pip install intel-extension-for-pytorch, and need to install intel_extension_for_pytorch for GPU, 
https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v1.10.200%2bgpu&os=linux&package=pip

python -m pip install torch==1.10.0a0 intel-extension-for-pytorch==1.10.200+gpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
python -m pip install torchvision==0.11.0+cpu --no-deps --index-url https://download.pytorch.org/whl/cpu (probably no need this)

2.Migrate quant_cuda_kernel.cu to SYCL: 

(sqllm) ayu1@ortce-a100-80G2:~/SqueezeLLM/SqueezeLLM$ dpct --in-root=squeezellm --out-root=output-sycl --cuda-include-path="/usr/local/cuda-12/include" --extra-arg="-I/nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/" --extra-arg="-I/nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages/torch/include/" --extra-arg="-I/usr/include/c++/11/" --extra-arg="-I/usr/include/x86_64-linux-gnu/c++/11/" --extra-arg="-I/nfs/site/home/ayu1/.conda/envs/sqllm/include/python3.9/" --extra-arg="-DCUDA_VERSION=12000" --process-all
Parsing: /nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu
Analyzing: /nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:263:7: warning: DPCT1038:0: When the kernel function name is used as a macro argument, the migration result may be incorrect. You need to verify the definition of the macro.
  263 |       SPMV_ATOMIC<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:263:7: warning: DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  263 |       SPMV_ATOMIC<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:309:7: warning: DPCT1038:2: When the kernel function name is used as a macro argument, the migration result may be incorrect. You need to verify the definition of the macro.
  309 |       SPMV_ATOMIC<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:309:7: warning: DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  309 |       SPMV_ATOMIC<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:369:7: warning: DPCT1038:4: When the kernel function name is used as a macro argument, the migration result may be incorrect. You need to verify the definition of the macro.
  369 |       SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:369:7: warning: DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  369 |       SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:423:7: warning: DPCT1038:6: When the kernel function name is used as a macro argument, the migration result may be incorrect. You need to verify the definition of the macro.
  423 |       SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:423:7: warning: DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  423 |       SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:476:7: warning: DPCT1038:8: When the kernel function name is used as a macro argument, the migration result may be incorrect. You need to verify the definition of the macro.
  476 |       SPMV_ATOMIC<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:476:7: warning: DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  476 |       SPMV_ATOMIC<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:547:7: warning: DPCT1038:10: When the kernel function name is used as a macro argument, the migration result may be incorrect. You need to verify the definition of the macro.
  547 |       SPMV_ATOMIC<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:547:7: warning: DPCT1049:11: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  547 |       SPMV_ATOMIC<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:620:7: warning: DPCT1038:12: When the kernel function name is used as a macro argument, the migration result may be incorrect. You need to verify the definition of the macro.
  620 |       SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:620:7: warning: DPCT1049:13: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  620 |       SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:701:7: warning: DPCT1038:14: When the kernel function name is used as a macro argument, the migration result may be incorrect. You need to verify the definition of the macro.
  701 |       SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:701:7: warning: DPCT1049:15: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  701 |       SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
      |       ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:922:5: warning: DPCT1118:16: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
  922 |     __syncthreads();
      |     ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:924:5: warning: DPCT1118:17: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
  924 |     __syncthreads();
      |     ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:1016:5: warning: DPCT1118:18: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
 1016 |     __syncthreads();
      |     ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:1018:5: warning: DPCT1118:19: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
 1018 |     __syncthreads();
      |     ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:1149:5: warning: DPCT1118:20: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
 1149 |     __syncthreads();
      |     ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:1151:5: warning: DPCT1118:21: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
 1151 |     __syncthreads();
      |     ^
Migrating: /nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:774:3: warning: DPCT1065:22: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
  774 |   __syncthreads();
      |   ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:855:3: warning: DPCT1065:23: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
  855 |   __syncthreads();
      |   ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:922:5: warning: DPCT1065:24: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
  922 |     __syncthreads();
      |     ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:924:5: warning: DPCT1065:25: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
  924 |     __syncthreads();
      |     ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:1016:5: warning: DPCT1065:26: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
 1016 |     __syncthreads();
      |     ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:1018:5: warning: DPCT1065:27: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
 1018 |     __syncthreads();
      |     ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:1107:3: warning: DPCT1065:28: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
 1107 |   __syncthreads();
      |   ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:1149:5: warning: DPCT1065:29: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
 1149 |     __syncthreads();
      |     ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:1151:5: warning: DPCT1065:30: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
 1151 |     __syncthreads();
      |     ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:753:20: warning: DPCT1101:31: 'BLOCKWIDTH' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
  753 |   __shared__ float blockvec[BLOCKWIDTH];
      |                    ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:757:20: warning: DPCT1101:32: 'BLOCKWIDTH' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
  757 |   __shared__ float deq2[8][BLOCKWIDTH];
      |                    ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:843:20: warning: DPCT1101:33: 'BLOCKWIDTH' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
  843 |   __shared__ float blockvec[BLOCKWIDTH];
      |                    ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:847:20: warning: DPCT1101:34: 'BLOCKWIDTH' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
  847 |   __shared__ float deq2[16][BLOCKWIDTH];
      |                    ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:898:20: warning: DPCT1101:35: 'BLOCKWIDTH' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
  898 |   __shared__ float blockvec[BLOCKWIDTH];
      |                    ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:900:20: warning: DPCT1101:36: 'BLOCKWIDTH' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
  900 |   __shared__ float deq2[8][BLOCKWIDTH];
      |                    ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:995:20: warning: DPCT1101:37: 'BLOCKWIDTH' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
  995 |   __shared__ float blockvec[BLOCKWIDTH];
      |                    ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:998:20: warning: DPCT1101:38: 'BLOCKWIDTH' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
  998 |   __shared__ float deq2[16][BLOCKWIDTH];
      |                    ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:1104:20: warning: DPCT1101:49: 'BLOCKWIDTH' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
 1104 |   __shared__ float blockvec[BLOCKWIDTH];
      |                    ^
/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm/quant_cuda_kernel.cu:1142:20: warning: DPCT1101:55: 'BLOCKWIDTH' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
 1142 |   __shared__ float blockvec[BLOCKWIDTH];
      |                    ^
Processed 1 file(s) in -in-root folder "/nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM/squeezellm"

See Diagnostics Reference to resolve warnings and complete the migration:
https://www.intel.com/content/www/us/en/docs/dpcpp-compatibility-tool/developer-guide-reference/current/diagnostics-reference.html

Also had to the type mismatching error, we changed most types to "auto" 

2. Create setup_sycl.py, get a Intel GPU machine and run it: python setup_sycl.py install

3. Due to running into an error: ValueError: invalid literal for int() with base 10: '0git'
We changed /nfs/site/home/ayu1/.local/lib/python3.9/site-packages/intel_extension_for_pytorch/xpu/cpp_extension.py
line 668: 
version = versionstr.decode(*SUBPROCESS_DECODE_ARGS).strip().split(".")
To 
version = versionstr.decode(*SUBPROCESS_DECODE_ARGS).strip().split("git")
version = version[0].split(".")

4. Modify quant_cuda.cpp
Change every const at::cuda::OptionalCUDAGuard device_guard(device_of(vec)); to const c10::OptionalDeviceGuard device_guard(device_of(vec)); in quant_cuda.cpp

5. Modify quant.py

6. Modify llama.py

7. Remember to export network proxy before running the model.
8. Run the model: 
(sqllm) ayu1@sdp4451:~/SqueezeLLM/SqueezeLLM$ SYCL_VISIBLE_DEVICES=0 python llama.py models/xgen-7b-8k-base c4 --wbits 3 --load sq-xgen-7b-8k-base-w3-s0.pt --benchmark 128 --check --torch_profile
models/xgen-7b-8k-base
Loading model ...
Done.
2024-07-26 23:07:04,470 - datasets - INFO - PyTorch version 2.1.0.post0+cxx11.abi available.
Using unk_token, but it is not set yet.
Using unk_token, but it is not set yet.
STAGE:2024-07-26 23:07:21 2076790:2076790 XPUActivityProfilerController.cpp:271] Completed Stage: Warm Up
Model type : llama
Benchmarking ...
0 3.416949510574341
1 0.12187480926513672
2 0.03214216232299805
3 0.032025814056396484
4 0.0318453311920166
5 0.031368255615234375
6 0.03156423568725586
7 0.03144264221191406
8 0.031032800674438477
9 0.03109455108642578
10 0.031516075134277344
11 0.03161191940307617
12 0.04979348182678223
13 0.031290292739868164
14 0.03127431869506836
15 0.03149294853210449
16 0.031514644622802734
17 0.030979394912719727
18 0.03138446807861328
19 0.031149625778198242
20 0.03110814094543457
21 0.031461477279663086
22 0.03165388107299805
23 0.0314326286315918
24 0.030878782272338867
25 0.03181624412536621
26 0.03124260902404785
27 0.032620906829833984
28 0.03110980987548828
29 0.030872821807861328
30 0.03081488609313965
31 0.03135251998901367
32 0.03246498107910156
33 0.030818700790405273
34 0.03099822998046875
35 0.03114485740661621
36 0.030895709991455078
37 0.03137326240539551
38 0.031003475189208984
39 0.030958175659179688
40 0.030821800231933594
41 0.03162503242492676
42 0.03426194190979004
43 0.03128457069396973
44 0.0308990478515625
45 0.030959367752075195
46 0.031003475189208984
47 0.031015396118164062
48 0.030828237533569336
49 0.03083205223083496
50 0.030695676803588867
51 0.031100988388061523
52 0.030643701553344727
53 0.031119585037231445
54 0.030945301055908203
55 0.03127264976501465
56 0.03075551986694336
57 0.03085184097290039
58 0.031186580657958984
59 0.031235694885253906
60 0.03082132339477539
61 0.030908584594726562
62 0.03166556358337402
63 0.030915498733520508
64 0.030939817428588867
65 0.030844688415527344
66 0.030871868133544922
67 0.030999422073364258
68 0.030856847763061523
69 0.031126976013183594
70 0.030744552612304688
71 0.03092670440673828
72 0.03075885772705078
73 0.030991554260253906
74 0.031056880950927734
75 0.03268027305603027
76 0.0308835506439209
77 0.03074336051940918
78 0.03119063377380371
79 0.03080272674560547
80 0.03066849708557129
81 0.030933380126953125
82 0.03109264373779297
83 0.031047582626342773
84 0.03112339973449707
85 0.030689001083374023
86 0.03077077865600586
87 0.03128170967102051
88 0.03072810173034668
89 0.03133058547973633
90 0.030883073806762695
91 0.030854225158691406
92 0.030648231506347656
93 0.03110051155090332
94 0.03099513053894043
95 0.030779361724853516
96 0.030684947967529297
97 0.0306699275970459
98 0.030907154083251953
99 0.030857086181640625
100 0.031004905700683594
101 0.030907154083251953
102 0.03091120719909668
103 0.030745267868041992
104 0.031088829040527344
105 0.03078913688659668
106 0.030786991119384766
107 0.030738115310668945
108 0.03070545196533203
109 0.031618356704711914
110 0.030683040618896484
111 0.03080892562866211
112 0.03060150146484375
113 0.0306699275970459
114 0.030868053436279297
115 0.030518531799316406
116 0.03119826316833496
117 0.0307924747467041
118 0.030636310577392578
119 0.031063318252563477
120 0.03226947784423828
121 0.031004667282104492
122 0.030728578567504883
123 0.031091928482055664
124 0.030841827392578125
125 0.031136512756347656
126 0.030635356903076172
127 0.10741925239562988
Median: 0.030998826026916504
PPL: 6488.31787109375
STAGE:2024-07-26 23:07:29 2076790:2076790 XPUActivityProfilerController.cpp:278] Completed Stage: Collection
STAGE:2024-07-26 23:07:30 2076790:2076790 XPUActivityProfilerController.cpp:281] Completed Stage: Post Processing
--------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                  Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
--------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                              aten::to         0.01%     642.000us         0.23%      12.240ms       0.118us        103677
                        aten::_to_copy         0.01%     520.000us         0.22%      12.073ms      93.589us           129
                   aten::empty_strided         0.33%      17.897ms         0.33%      17.897ms       1.084us         16513
                           aten::copy_         0.52%      28.015ms         0.52%      28.015ms     109.008us           257
                            aten::ones         0.00%      13.000us         0.06%       3.386ms       3.386ms             1
                           aten::empty         0.45%      24.414ms         0.45%      24.414ms       0.141us        173697
                           aten::fill_         0.06%       3.369ms         0.06%       3.369ms       3.369ms             1
                           aten::slice         0.72%      38.676ms         0.72%      38.962ms       0.669us         58238
                      aten::as_strided         0.01%     739.000us         0.01%     739.000us       0.006us        128764
                         aten::reshape         0.59%      32.097ms         0.69%      37.538ms       0.650us         57728
                            aten::view         0.05%       2.530ms         0.05%       2.530ms       0.038us         66208
                          aten::arange         0.07%       3.825ms         0.13%       7.219ms      28.199us           256
                         aten::resize_         0.35%      19.128ms         0.35%      19.128ms       0.761us         25150
                       aten::unsqueeze         0.10%       5.561ms         0.10%       5.596ms       0.653us          8576
                       aten::embedding         0.01%     604.000us         1.72%      92.835ms     725.273us           128
                    aten::index_select         1.70%      92.068ms         1.71%      92.205ms     720.352us           128
                          aten::expand         0.19%      10.178ms         0.19%      10.242ms       0.620us         16512
                            aten::rsub         0.01%     530.000us         0.30%      16.444ms     128.469us           128
                             aten::sub         0.29%      15.779ms         0.57%      30.917ms     120.770us           256
                     aten::masked_fill         0.01%     389.000us         0.35%      18.970ms     148.203us           128
                           aten::clone         0.01%     385.000us         0.32%      17.195ms     134.336us           128
                      aten::empty_like         0.00%      83.000us         0.00%     168.000us       1.312us           128
                    aten::masked_fill_         0.02%       1.307ms         0.02%       1.307ms      10.211us           128
                             aten::pow         2.04%     110.461ms         3.47%     187.630ms      11.276us         16640
                     aten::result_type         0.00%       8.000us         0.00%       8.000us       0.000us         16640
                        aten::can_cast         0.00%       0.000us         0.00%       0.000us       0.000us          8320
                            aten::mean         2.21%     119.452ms         2.21%     119.701ms      14.387us          8320
                             aten::add         4.91%     265.499ms         4.94%     267.194ms       9.277us         28801
                           aten::rsqrt         1.49%      80.347ms         2.53%     137.024ms       8.235us         16640
                             aten::mul         5.44%     294.317ms         5.70%     308.423ms       8.309us         37120
                           aten::zeros         1.29%      69.474ms         4.14%     223.981ms       7.812us         28672
                           aten::zero_         3.27%     176.887ms         3.27%     176.887ms       6.169us         28672
                     clGetPlatformInfo         0.00%      29.000us         0.00%      29.000us       0.169us           172
                        clGetDeviceIDs         0.00%      19.000us         0.00%      19.000us       0.322us            59
                       clGetDeviceInfo         0.00%      20.000us         0.00%      20.000us       0.041us           484
                        clRetainDevice         0.00%       0.000us         0.00%       0.000us       0.000us            36
                       clReleaseDevice         0.00%       0.000us         0.00%       0.000us       0.000us            61
                       clCreateContext         2.21%     119.577ms         2.21%     119.577ms      11.958ms            10
    clCreateCommandQueueWithProperties         0.00%      69.000us         0.00%      69.000us       9.857us             7
                       aten::transpose         0.28%      15.117ms         0.28%      15.265ms       0.741us         20608
                         aten::squeeze         0.18%       9.641ms         0.18%       9.698ms       0.592us         16384
                           aten::index         2.50%     135.120ms         2.58%     139.298ms      17.004us          8192
                             aten::neg         3.90%     210.650ms         7.46%     403.347ms      24.618us         16384
                             aten::cat         8.82%     476.708ms         9.52%     514.812ms      31.545us         16320
                          aten::narrow         0.38%      20.659ms         0.39%      21.325ms       1.302us         16384
                          aten::matmul         0.89%      48.369ms        32.11%        1.736s     208.650us          8320
                             aten::bmm        26.70%        1.443s        29.97%        1.620s     197.805us          8192
                      clGetContextInfo         0.00%       0.000us         0.00%       0.000us       0.000us            12
             clCreateProgramWithSource         0.00%       4.000us         0.00%       4.000us       4.000us             1
                        clBuildProgram         3.22%     173.908ms         3.22%     173.908ms      57.969ms             3
                      clGetProgramInfo         0.00%       1.000us         0.00%       1.000us       0.250us             4
                      clReleaseProgram         0.00%      27.000us         0.00%      27.000us       9.000us             3
             clCreateProgramWithBinary         0.00%      38.000us         0.00%      38.000us      19.000us             2
                        clCreateKernel         0.00%       3.000us         0.00%       3.000us       3.000us             1
                       clGetKernelInfo         0.00%       0.000us         0.00%       0.000us       0.000us             1
                    clGetKernelArgInfo         0.00%      12.000us         0.00%      12.000us       1.500us             8
                        clRetainKernel         0.00%       0.000us         0.00%       0.000us       0.000us             1
                       clReleaseKernel         0.00%       5.000us         0.00%       5.000us       2.500us             2
                      clReleaseContext         0.00%      23.000us         0.00%      23.000us       2.091us            11
                 clReleaseCommandQueue         0.01%     283.000us         0.01%     283.000us     283.000us             1
                    clCreateSubDevices         0.00%       0.000us         0.00%       0.000us       0.000us             2
                       clRetainContext         0.00%       0.000us         0.00%       0.000us       0.000us             4
                    aten::_unsafe_view         0.01%     291.000us         0.01%     291.000us       0.035us          8320
                             aten::div         3.88%     209.900ms         3.94%     213.102ms      52.014us          4097
                      aten::lift_fresh         0.00%       0.000us         0.00%       0.000us       0.000us          4096
                         aten::detach_         0.08%       4.212ms         0.08%       4.223ms       1.031us          4096
                               detach_         0.00%      12.000us         0.00%      12.000us       0.003us          4096
                             aten::max         0.19%      10.132ms         3.71%     200.501ms      48.950us          4096
                         aten::maximum         3.65%     197.323ms         3.65%     197.392ms      48.191us          4096
                         aten::softmax         0.25%      13.363ms         4.65%     251.401ms      61.377us          4096
                        aten::_softmax         4.54%     245.358ms         8.77%     474.301ms      57.912us          8190
                            aten::silu         4.03%     217.746ms         7.84%     423.781ms      51.731us          8192
                          aten::linear         0.02%     877.000us         0.65%      35.005ms     273.477us           128
                               aten::t         0.00%     250.000us         0.01%     419.000us       3.273us           128
                              aten::mm         0.62%      33.281ms         0.62%      33.743ms     263.617us           128
                          aten::select         0.01%     662.000us         0.01%     717.000us       2.823us           254
              aten::cross_entropy_loss         0.05%       2.561ms         4.93%     266.568ms       2.099ms           127
                     aten::log_softmax         0.02%     844.000us         1.73%      93.524ms     736.409us           127
                    aten::_log_softmax         1.71%      92.587ms         3.43%     185.380ms     729.843us           254
                     aten::nll_loss_nd         0.00%     251.000us         3.14%     169.839ms       1.337ms           127
                        aten::nll_loss         0.02%       1.041ms         3.14%     169.658ms       1.336ms           127
                aten::nll_loss_forward         3.11%     168.401ms         6.22%     336.451ms       1.325ms           254
                  aten::_reshape_alias         0.08%       4.113ms         0.08%       4.113ms       1.012us          4064
                            aten::add_         0.25%      13.505ms         0.25%      13.505ms     107.183us           126
                             aten::exp         2.24%     120.901ms         4.07%     220.305ms     110.153ms             2
                            aten::item         0.00%       5.000us         0.00%      47.000us      47.000us             1
             aten::_local_scalar_dense         0.00%      42.000us         0.00%      42.000us      42.000us             1
--------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 5.407s


Talked to IPEX team to understand how to profile Intel GPU, had to install this: 
https://intel.github.io/intel-extension-for-pytorch/#installation?platform=gpu
IPEX team also suggested to upgrade IPEX with these: 
https://ubit-artifactory-sh.intel.com/artifactory/aipc_releases-sh-local/gpu-new/releases/2024.2/IPEX_2.3.110+xpu/RC1/
Checked Python 3.9, so used py3.9 link: (If met network issue, do this: export no_proxy=ubit-artifactory-sh.intel.com)
wget https://ubit-artifactory-sh.intel.com/artifactory/aipc_releases-sh-local/gpu-new/releases/2024.2/IPEX_2.3.110+xpu/RC1/py39/intel_extension_for_pytorch-2.3.110+xpu-cp39-cp39-linux_x86_64.whl
wget https://ubit-artifactory-sh.intel.com/artifactory/aipc_releases-sh-local/gpu-new/releases/2024.2/IPEX_2.3.110+xpu/RC1/py39/oneccl_bind_pt-2.3.100+xpu-cp39-cp39-linux_x86_64.whl
wget https://ubit-artifactory-sh.intel.com/artifactory/aipc_releases-sh-local/gpu-new/releases/2024.2/IPEX_2.3.110+xpu/RC1/py39/torch-2.3.1+cxx11.abi-cp39-cp39-linux_x86_64.whl
wget https://ubit-artifactory-sh.intel.com/artifactory/aipc_releases-sh-local/gpu-new/releases/2024.2/IPEX_2.3.110+xpu/RC1/py39/torchaudio-2.3.1+cxx11.abi-cp39-cp39-linux_x86_64.whl
Then pip install everything. 

Current pip list: 
Package                     Version          Editable project location
--------------------------- ---------------- ---------------------------------------------------------------------------------------------------------
accelerate                  0.31.0
aiohttp                     3.9.5
aiosignal                   1.3.1
annotated-types             0.7.0
async-timeout               4.0.3
attrs                       23.2.0
certifi                     2024.6.2
charset-normalizer          3.3.2
datasets                    2.19.2
dill                        0.3.8
et-xmlfile                  1.1.0
filelock                    3.14.0
frozenlist                  1.4.1
fsspec                      2024.3.1
huggingface-hub             0.23.3
idna                        3.7
intel_extension_for_pytorch 2.3.110+xpu
Jinja2                      3.1.4
MarkupSafe                  2.1.5
mpmath                      1.3.0
multidict                   6.0.5
multiprocess                0.70.16
networkx                    3.2.1
ninja                       1.10.2.3
numpy                       1.26.4
nvidia-cublas-cu12          12.1.3.1
nvidia-cuda-cupti-cu12      12.1.105
nvidia-cuda-nvrtc-cu12      12.1.105
nvidia-cuda-runtime-cu12    12.1.105
nvidia-cudnn-cu12           8.9.2.26
nvidia-cufft-cu12           11.0.2.54
nvidia-curand-cu12          10.3.2.106
nvidia-cusolver-cu12        11.4.5.107
nvidia-cusparse-cu12        12.1.0.106
nvidia-nccl-cu12            2.20.5
nvidia-nvjitlink-cu12       12.5.40
nvidia-nvtx-cu12            12.1.105
oneccl-bind-pt              2.3.100+xpu
openpyxl                    3.1.2
packaging                   24.1
pandas                      2.2.2
pillow                      10.4.0
pip                         23.1.2
popcorn                     0.0.2
prettytable                 3.9.0
psutil                      6.0.0
pyarrow                     16.1.0
pyarrow-hotfix              0.6
pydantic                    2.8.2
pydantic_core               2.20.1
python-dateutil             2.9.0.post0
pytz                        2024.1
PyYAML                      6.0.1
quant-cuda                  0.0.0
quant_sycl                  0.0.0            /nfs/site/home/ayu1/.conda/envs/sqllm/lib/python3.9/site-packages/quant_sycl-0.0.0-py3.9-linux-x86_64.egg
regex                       2024.5.15
requests                    2.32.3
ruamel.yaml                 0.18.6
ruamel.yaml.clib            0.2.8
safetensors                 0.4.3
sentencepiece               0.2.0
setuptools                  69.5.1
six                         1.16.0
squeezellm                  0.1.0            /nfs/site/home/ayu1/SqueezeLLM/SqueezeLLM
sympy                       1.12.1
tiktoken                    0.7.0
tokenizers                  0.13.3
torch                       2.3.1+cxx11.abi
torchaudio                  2.3.1+cxx11.abi
torchvision                 0.18.1+cxx11.abi
tqdm                        4.66.4
transformers                4.29.0
triton                      2.3.1
typing_extensions           4.12.2
tzdata                      2024.1
urllib3                     2.2.1
wcwidth                     0.2.13
wheel                       0.40.0
xxhash                      3.4.1
yarl                        1.9.4

Also need to do install pti: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-5.html
Check "Option 2D: Install Using Offline Installation Scripts" section in the page: 
cd /tmp
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/884eaa22-d56f-45dc-9a65-901f1c625f9e/l_intel-for-pytorch-gpu-dev_p_0.5.3.36_offline.sh
sh ./l_intel-for-pytorch-gpu-dev_p_0.5.3.36_offline.sh
Since may not be able to install it in /opt/intel/oneapi, can install in ~/intel/oneapi instead, and source ~/intel/oneapi/pti/latest/env/vars.sh

There's a known issue with not finding the library, so need to do this: 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33
(it might be another version libstdc++.so.6.0.xx, need to check which one it is under /usr/lib/x86_64-linux-gnu/)


