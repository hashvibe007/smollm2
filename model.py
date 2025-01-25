import os
import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)

from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.utils.rnn import pad_sequence
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import TQDMProgressBar

# Set environment variable for memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



# Function to log GPU memory usage
def log_memory_usage(step):
    if torch.cuda.is_available():
        print(
            f"Step {step}: "
            f"Allocated = {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
            f"Reserved = {torch.cuda.memory_reserved() / 1e9:.2f} GB"
        )


# Custom Collate Function
def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = pad_sequence(
        labels, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    return {"input_ids": input_ids, "labels": labels}


# Streaming Dataset
class StreamingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for example in iter(self.dataset):
            tokenized = self.tokenizer(
                example["text"],
                truncation=True,
                max_length=self.max_length,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )
            for chunk in tokenized["input_ids"]:
                yield {
                    "input_ids": chunk.squeeze(0),
                    "labels": chunk.squeeze(0),
                }


# Lightning Module
class SmolLMModule(LightningModule):
    def __init__(self, config, learning_rate=1e-4):
        super().__init__()
        self.config = config
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        model_config = LlamaConfig(
            vocab_size=49152,
            hidden_size=config["model"]["model_config"]["hidden_size"],
            intermediate_size=config["model"]["model_config"]["intermediate_size"],
            num_hidden_layers=config["model"]["model_config"]["num_hidden_layers"],
            num_attention_heads=config["model"]["model_config"]["num_attention_heads"],
            num_key_value_heads=config["model"]["model_config"]["num_key_value_heads"],
            hidden_act=config["model"]["model_config"]["hidden_act"],
            max_position_embeddings=config["model"]["model_config"][
                "max_position_embeddings"
            ],
            initializer_range=config["model"]["model_config"]["initializer_range"],
            rms_norm_eps=1e-5,
            use_cache=True,
            pad_token_id=config["model"]["model_config"]["pad_token_id"],
            bos_token_id=config["model"]["model_config"]["bos_token_id"],
            eos_token_id=config["model"]["model_config"]["eos_token_id"],
        )
        self.model = AutoModelForCausalLM.from_config(model_config)

    def training_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
        loss = outputs.loss
        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True
        )  # Log loss

        # # Log memory usage
        # if batch_idx % 10 == 0:
        #     log_memory_usage(batch_idx)

        # Release intermediate tensors
        del outputs
        torch.cuda.empty_cache()

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=self.config["optimizer"]["weight_decay"],
        )


# Main Script
if __name__ == "__main__":
    # Load config
    with open("/kaggle/input/yaml-file/config_smollm2_135.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True
    )
    train_dataset = dataset["train"]

    # Create DataLoader
    streaming_dataset = StreamingDataset(train_dataset, tokenizer, max_length=2048)
    train_loader = DataLoader(
        streaming_dataset,
        batch_size=1,  # Reduced batch size
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model
    model = SmolLMModule(
        config,
        learning_rate=config["optimizer"]["learning_rate_scheduler"]["learning_rate"],
    )

    # Initialize logger with version based on start_step
    logger = TensorBoardLogger("logs", name="smollm2")

    # Checkpoint callback configuration
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="model-{epoch:02d}-{step}-{train_loss:.2f}",  # Include training loss in filename
        monitor="train_loss",  # Monitor training loss
        mode="min",  # Lower loss is better
        save_top_k=3,  # Save the best 3 models
        save_last=True,  # Additionally save the last model
        every_n_train_steps=5000,  # Save every 500 steps
        save_weights_only=False,  # Save the full model state
        auto_insert_metric_name=False,  # Don't insert metric name in filename
    )

    # Progress bar
    # progress_bar = RichProgressBar(
    #     refresh_rate=1,
    #     leave=False,
    #     theme=RichProgressBarTheme(
    #         description="",
    #         progress_bar="#6206E0",
    #         progress_bar_finished="#6206E0",
    #         progress_bar_pulse="#6206E0",
    #         batch_progress="",
    #         time="dim",
    #         processing_speed="dim underline",
    #         metrics="italic",
    #         metrics_text_delimiter=" ",
    #         metrics_format=".3f",
    #     ),
    #     console_kwargs=None,
    # )
    progress_bar = TQDMProgressBar(refresh_rate=10)

    # Create trainer
    trainer = Trainer(
        logger=logger,
        strategy="ddp_notebook",
        accelerator="gpu",
        devices=2,
        precision="16-mixed",
        max_steps=500000,
        accumulate_grad_batches=1,
        enable_checkpointing = True,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            progress_bar,
            checkpoint_callback,
        ],
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
    )

    # Find latest checkpoint if exists
    if os.path.exists("checkpoints/last.ckpt"):
        resume_from_checkpoint = "checkpoints/last.ckpt"
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    else:
        resume_from_checkpoint = None
        print("Starting training from scratch")

    # Train with automatic checkpoint resumption
    trainer.fit(model, train_loader, ckpt_path=resume_from_checkpoint)
    optimizers = trainer.optimizers
    if optimizers:
        optimizer = optimizers[0]
        print("optimizer state:",optimizer.state_dict())

    # After training, print the best model path and score
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    # print(f"Best train loss: {checkpoint_callback.best_model_score:.4f}")

    # Save final model
    if trainer.is_global_zero:
        output_dir = "final_model"
        os.makedirs(output_dir, exist_ok=True)
        model.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
