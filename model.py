# import libraries
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
from transformers import Trainer
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader

# load dataset
dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True)
train_dataset = dataset["train"]
for sample in train_dataset:
    print(sample)
    break
# load tokenizer
# use tokeniser from https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
# Set padding token to be the same as EOS token
tokenizer.pad_token = tokenizer.eos_token

# load config
# use config from https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config_smollm2_135M.yaml
# config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")


def collate_fn(examples):
    # Tokenize the texts
    encoding = tokenizer(
        [example["text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # Create labels (same as input_ids for causal language modeling)
    encoding["labels"] = encoding["input_ids"].clone()

    return encoding


def create_model_config(config):
    model_config = config["model"]["model_config"]
    return LlamaConfig(
        vocab_size=49152,  # From the model architecture
        hidden_size=model_config["hidden_size"],
        intermediate_size=model_config["intermediate_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        num_key_value_heads=model_config["num_key_value_heads"],
        hidden_act=model_config["hidden_act"],
        max_position_embeddings=model_config["max_position_embeddings"],
        initializer_range=model_config["initializer_range"],
        rms_norm_eps=1e-5,  # From the model architecture
        use_cache=True,
        pad_token_id=model_config["pad_token_id"],
        bos_token_id=model_config["bos_token_id"],
        eos_token_id=model_config["eos_token_id"],
    )


# create model
class SmolLMModule(pl.LightningModule):
    def __init__(self, config, learning_rate=1e-4):
        super().__init__()
        self.config = config
        self.learning_rate = learning_rate
        self.save_hyperparameters()  # Save hyperparameters for resuming

        # Create model from config
        model_config = create_model_config(config)
        self.model = AutoModelForCausalLM.from_config(model_config)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        )
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        # Save additional info if needed
        checkpoint["step"] = self.global_step
        checkpoint["model_config"] = self.config

    def on_load_checkpoint(self, checkpoint):
        # Restore additional info if needed
        self.global_step = checkpoint["step"]
        self.config = checkpoint["model_config"]


# train model

# save model

# training script
if __name__ == "__main__":
    import os
    from pytorch_lightning.callbacks import ModelCheckpoint

    # parameters load from config file
    with open("config_smollm2_135.yaml", "r") as file:
        config = yaml.safe_load(file)
    max_steps = 5000  # Total training steps

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model-step={step}",
        save_top_k=-1,  # Save all checkpoints
        every_n_train_steps=500,  # Save every 500 steps
        save_weights_only=False,  # Save the full model state
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    # Set padding token to be the same as EOS token
    tokenizer.pad_token = tokenizer.eos_token

    # load dataset
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True
    )
    train_dataset = dataset["train"]

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Small batch size for testing
        collate_fn=collate_fn,
        num_workers=2,
    )

    # create model
    model = SmolLMModule(config, learning_rate=1e-4)

    # progress bar
    progress_bar = RichProgressBar(leave=False, refresh_rate=1, console_kwargs=None)

    # Find latest checkpoint if exists
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
        if checkpoints:
            # Sort by step number and get the latest
            latest_checkpoint = os.path.join(
                checkpoint_dir,
                sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))[
                    -1
                ],
            )
            print(f"Resuming from checkpoint: {latest_checkpoint}")

    # create trainer
    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            progress_bar,
            checkpoint_callback,
        ],
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # train model
    if latest_checkpoint:
        # Resume training from checkpoint if it exists
        trainer.fit(model, train_loader, ckpt_path=latest_checkpoint)
    else:
        # Start training from scratch
        trainer.fit(model, train_loader)

    # Save final model and tokenizer
    if trainer.is_global_zero:  # Only save on main process
        output_dir = "final_model"
        os.makedirs(output_dir, exist_ok=True)
        model.model.save_pretrained(os.path.join(output_dir, "model"))
        tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
