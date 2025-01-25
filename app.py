import os
import gradio as gr
import torch
from model import SmolLMModule
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import glob

# Load config
with open("config_smollm2_135.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
tokenizer.pad_token = tokenizer.eos_token


def get_available_checkpoints():
    """Get list of available checkpoints and final model"""
    models = []
    model_paths = {}

    # Get checkpoints
    checkpoints = glob.glob("checkpoints/*.ckpt")
    for ckpt in checkpoints:
        try:
            # Extract step number from the filename
            filename = os.path.basename(ckpt)
            # Handle the format 'model-step=step=X.ckpt'
            if "step=step=" in filename:
                step = int(filename.split("step=step=")[1].split(".")[0])
                display_name = f"Checkpoint Step {step}"
                models.append(display_name)
                model_paths[display_name] = ckpt
        except (ValueError, IndexError) as e:
            print(
                f"Warning: Could not parse checkpoint filename: {filename}, Error: {e}"
            )
            continue

    # Add final model if it exists
    final_model_path = "final_model"
    if os.path.exists(final_model_path):
        display_name = "Final Model"
        models.append(display_name)
        model_paths[display_name] = final_model_path

    # Sort checkpoints by step number (Final model will be at the end)
    def get_step_number(name):
        if name == "Final Model":
            return float("inf")
        try:
            return int(name.split("Step ")[-1])
        except:
            return 0

    models.sort(key=get_step_number)

    if not models:
        print(
            "Warning: No checkpoints or final model found in the following locations:"
        )
        print("- Checkpoints directory:", os.path.abspath("checkpoints"))
        print("- Final model directory:", os.path.abspath("final_model"))
    else:
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"- {model}: {model_paths[model]}")

    return models, model_paths


def load_model_from_checkpoint(model_path):
    """Load model from checkpoint or final model directory"""
    if model_path == "final_model":
        # Load the final saved model
        model = SmolLMModule(config)
        model.model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        # Load from checkpoint
        model = SmolLMModule.load_from_checkpoint(model_path, config=config)

    model.eval()  # Set to evaluation mode
    return model


def generate_text(prompt, model_choice, max_length=100, temperature=0.7, top_p=0.9):
    """Generate text based on prompt using selected model"""
    # Check if model is selected
    if not model_choice:
        return "Please select a model checkpoint!"

    if not prompt:
        return "Please enter a prompt!"

    try:
        # Get model path from the mapping
        _, model_paths = get_available_checkpoints()
        model_path = model_paths.get(model_choice)

        if not model_path or not os.path.exists(model_path):
            return f"Model {model_choice} not found!"

        # Load model
        model = load_model_from_checkpoint(model_path)

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


# Get available models
display_names, _ = get_available_checkpoints()

# Create Gradio interface
with gr.Blocks(title="SmolLM2 Inference") as demo:
    gr.Markdown("# SmolLM2 Text Generation")

    if not display_names:
        gr.Markdown("⚠️ No models found! Please train the model first.")
    else:
        gr.Markdown(
            f"Found {len(display_names)} models/checkpoints. Select one and enter a prompt to generate text."
        )
        gr.Markdown("Available models: " + ", ".join(display_names))

    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=display_names,
                label="Select Model",
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
        inputs=[prompt, model_dropdown, max_length, temperature, top_p],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch(share=True)
