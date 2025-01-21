import os
import gradio as gr
import torch
from model import SmolLMModule, create_model_config
from transformers import AutoTokenizer
import yaml
import glob

# Load config
with open("config_smollm2_135.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
tokenizer.pad_token = tokenizer.eos_token


def load_model_from_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    model = SmolLMModule.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()  # Set to evaluation mode
    return model


def get_available_checkpoints():
    """Get list of available checkpoints sorted by step number"""
    checkpoints = glob.glob("checkpoints/*.ckpt")
    if not checkpoints:
        return [], []

    # Sort by step number
    def get_step_number(filepath):
        try:
            # Extract step number from the filename
            filename = os.path.basename(filepath)
            # Remove .ckpt extension
            filename = filename.replace(".ckpt", "")
            # Get the step number
            if "step=" in filename:
                return int(filename.split("step=")[1])
            elif "-step-" in filename:
                return int(filename.split("-step-")[1])
            else:
                return int("".join(filter(str.isdigit, filename)))
        except (ValueError, IndexError):
            return 0

    # Sort checkpoints by step number
    checkpoints.sort(key=get_step_number)

    # Create display names
    display_names = [f"Step {get_step_number(x)}" for x in checkpoints]
    return display_names, checkpoints


def generate_text(
    prompt, checkpoint_choice, max_length=100, temperature=0.7, top_p=0.9
):
    """Generate text based on prompt using selected checkpoint"""
    # Check if checkpoint is selected
    if not checkpoint_choice:
        return "Please select a checkpoint first!"

    if not prompt:
        return "Please enter a prompt!"

    try:
        # Get actual checkpoint path
        step_num = int("".join(filter(str.isdigit, checkpoint_choice)))
        checkpoints = glob.glob("checkpoints/*.ckpt")
        checkpoint_path = None

        for ckpt in checkpoints:
            if str(step_num) in ckpt:
                checkpoint_path = ckpt
                break

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return f"Checkpoint for step {step_num} not found!"

        # Load model from checkpoint
        model = load_model_from_checkpoint(checkpoint_path)

        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        # Move inputs to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode and return generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        return f"Error during generation: {str(e)}"


# Get available checkpoints
display_names, _ = get_available_checkpoints()

# Create Gradio interface
with gr.Blocks(title="SmolLM2 Inference") as demo:
    gr.Markdown("# SmolLM2 Text Generation")

    if not display_names:
        gr.Markdown("⚠️ No checkpoints found! Please train the model first.")
    else:
        gr.Markdown(
            f"Found {len(display_names)} checkpoints. Select one and enter a prompt to generate text."
        )

    with gr.Row():
        with gr.Column():
            checkpoint_dropdown = gr.Dropdown(
                choices=display_names,
                label="Select Checkpoint",
                value=display_names[-1] if display_names else None,
                interactive=True,
            )
            prompt = gr.Textbox(
                lines=3, placeholder="Enter your prompt here...", label="Input Prompt"
            )
            max_length = gr.Slider(
                minimum=10, maximum=500, value=100, step=10, label="Max Length"
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature"
            )
            top_p = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top-p"
            )
            generate_btn = gr.Button("Generate")

        with gr.Column():
            output = gr.Textbox(lines=8, label="Generated Text")

    generate_btn.click(
        fn=generate_text,
        inputs=[prompt, checkpoint_dropdown, max_length, temperature, top_p],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch(share=True)
