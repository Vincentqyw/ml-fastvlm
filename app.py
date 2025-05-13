import argparse
from pathlib import Path
import gradio as gr
import torch
from PIL import Image
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

# Available models
MODELS = {
    "FastViTHD 0.5BS2": "./checkpoints/llava-fastvithd_0.5b_stage2",
    "FastViTHD 0.5BS3": "./checkpoints/llava-fastvithd_0.5b_stage3",
    "FastViTHD 1.5BS3": "./checkpoints/llava-fastvithd_1.5b_stage3",
    # Add more models here as needed
}


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


device = get_device()


class ImageDescriber:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.current_model = None

    def load_model(self, model_path):
        if model_path == self.current_model:
            return

        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(model_path, None, model_name, device=device)
        )
        self.current_model = model_path

    def describe_image(
        self,
        image,
        prompt="Describe the image.",
        model_name="FastViTHD 0.5BS2",
        conv_mode="qwen_2",
        temperature=0.2,
        top_p=0.7,
        num_beams=1,
    ):
        try:
            # Load model if not loaded or changed
            model_path = MODELS[model_name]
            self.load_model(model_path)

            # Process image
            image = Image.fromarray(image)
            image_tensor = process_images(
                [image], self.image_processor, self.model.config
            )[0]

            # Construct prompt
            qs = prompt
            if self.model.config.mm_use_im_start_end:
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()

            # Tokenize and generate
            input_ids = (
                tokenizer_image_token(
                    prompt_text,
                    self.tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .to(device)
            )

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half(),
                    image_sizes=[image.size],
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=256,
                    use_cache=True,
                )

                description = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0].strip()
                print(f"Generated description: \n{description}")
                return description

        except Exception as e:
            return f"Error: {str(e)}"


# Create interface
describer = ImageDescriber()

with gr.Blocks() as demo:
    gr.Markdown("# FastVLM Image Description")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image")
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=list(MODELS.keys()),
                value="FastViTHD 0.5BS2",
            )
            prompt_input = gr.Textbox(
                label="Prompt", value="Describe the image"
            )
            with gr.Accordion("Generation Parameters", open=False):
                temp_slider = gr.Slider(
                    label="Temperature",
                    minimum=0,
                    maximum=1,
                    value=0.2,
                    step=0.1,
                )
                top_p_slider = gr.Slider(
                    label="Top-p", minimum=0, maximum=1, value=0.7, step=0.1
                )
                num_beams_input = gr.Number(
                    label="Num Beams", value=1, precision=0
                )

        with gr.Column():
            output_text = gr.Textbox(
                label="Description",
                interactive=False,
                # lines=10,
                max_lines=1000,
                show_copy_button=True,
                autoscroll=True,
            )
            run_button = gr.Button("Generate Description", variant="primary")

    # Add examples
    gr.Examples(
        examples=[["./test.jpg", "Describe the image"]],
        inputs=[image_input, prompt_input],
        outputs=output_text,
        fn=describer.describe_image,
        cache_examples=False,
    )

    run_button.click(
        fn=describer.describe_image,
        inputs=[
            image_input,
            prompt_input,
            model_dropdown,
            gr.Text("qwen_2", visible=False),  # conv_mode
            temp_slider,
            top_p_slider,
            num_beams_input,
        ],
        outputs=output_text,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="server name",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="server port",
    )
    args = parser.parse_args()
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=False,
        allowed_paths=[
            str(Path(__file__).parents[0]),
            str(Path(__file__).parents[1]),
        ],
    )
