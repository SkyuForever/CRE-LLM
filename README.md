# CRE-LLM
<h4 >CRE-LLM: A Domain-Specific Chinese Relation Extraction Framework with Fine-tuned Large Language Model</h4>

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://arxiv.org/abs/2310.08975)

\[ English | [中文](README_zh.md) \]

##  Overview 

![](./figs/Figure2.drawio.png)

## Supported Models

| Model                                                    | Model size                  | Default module    | Template  |
| -------------------------------------------------------- | --------------------------- | ----------------- | --------- |
| [LLaMA-2](https://huggingface.co/meta-llama)             | 7B/13B/70B                  | q_proj,v_proj     | llama2    |
| [Baichuan2](https://github.com/baichuan-inc/Baichuan2)   | 7B/13B                      | W_pack            | baichuan2 |
| [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)         | 6B                          | query_key_value   | chatglm2  |

## Requirement

- Python 3.8+ and PyTorch 1.13.1+
- 🤗Transformers, Datasets, Accelerate, PEFT and TRL
- sentencepiece, protobuf and tiktoken

And **powerful GPUs**!

## Getting Started

### Data Preparation (optional)

> [!NOTE]
> Please update `data/dataset_info.json` to use your custom dataset. About the format of this file, please refer to `data/README.md`.

### Dependence Installation (optional)

```bash
git clone https://github.com/SkyuForever/CRE-LLM.git
conda create -n CRE-LLM python=3.10
conda activate CRE-LLM
cd CRE-LLM
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

Here is an example of altering the self-cognition of an instruction-tuned language model within 10 minutes on a single GPU.

### Train on a single GPU

> [!IMPORTANT]

#### Supervised Fine-Tuning

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_model \
    --do_train \
    --dataset FinRE \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
```

### Predict

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_model \
    --do_predict \
    --dataset FinRE \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --predict_with_generate
```

> [!NOTE]
> We recommend using `--per_device_eval_batch_size=1` and `--max_target_length 128` at 4/8-bit predict.

## Citation

If this work is helpful, please kindly cite as:

```bibtex

```

## Acknowledgement

This repo benefits from [PEFT](https://github.com/huggingface/peft), [LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning), [SimCSE](https://github.com/princeton-nlp/SimCSE) and [DECAF](https://github.com/awslabs/decode-answer-logical-form). Thanks for their wonderful works.

