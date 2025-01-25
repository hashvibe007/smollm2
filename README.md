---
---
title: SmolLM2-135M
emoji: 🚀
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: "1.20.0"
app_file: app.py
pinned: false
---

---

<!-- use venv to create a virtual environment -->
```
uv venv 
source .venv/bin/activate
```
<!-- Train smollm2 model -->
use dataset from https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/main/cosmopedia-v2
```
dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2")
```

use tokeniser from https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer
```
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
```
use config from https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config_smollm2_135M.yaml

https://github.com/huggingface/smollm/blob/main/pre-training/smollm2/config_smollm2_135M.yaml

create model from above parameters

Use it for training using pytorch lightning 

<!-- Model architecture -->

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((576,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((576,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((576,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)