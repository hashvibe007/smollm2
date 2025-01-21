from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
# from datasets import load_dataset


# dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2")

# use tokeniser https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer
# tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")

print(model)
