# from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from peft import get_peft_model, LoraConfig, TaskType
from typing import Any, Optional, List
import torch
from transformers.modeling_utils import PreTrainedModel
from src.utils.logging import logger


def prepare_model_for_training(
    model: "PreTrainedModel",
    output_layer_name: Optional[str] = "lm_head",
    use_gradient_checkpointing: Optional[bool] = True,
    layer_norm_names: Optional[List[str]] = ["q_proj", "v_proj"],
) -> "PreTrainedModel":
    r"""
    Includes:
        (1) cast the layernorm in fp32
        (2) make output embedding layer require grads
        (3) upcast the lm_head to fp32
    Inspired by: https://github.com/huggingface/peft/blob/v0.2.0/src/peft/utils/other.py#L33
    """
    logger.info("prepare_model_for_training")
    for name, param in model.named_parameters():
        if param.ndim == 1 and any(
            layer_norm_name in name for layer_norm_name in layer_norm_names
        ):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()
        model.config.use_cache = (
            False  # turn off when gradient checkpointing is enabled
        )

    logger.info("CastOutputToFloat")
    if hasattr(model, output_layer_name):
        output_layer: torch.nn.Linear = getattr(model, output_layer_name)
        input_dtype = output_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_layer_name, CastOutputToFloat(output_layer))
    return model


def load_whisper_model(
    model_name_or_path: str,
    use_peft: bool = False,
    peft_config: dict = None,
    language: str = "chinese",
    compute_dtype: Any = None,
):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name_or_path
    )  # , torch_dtype=compute_dtype,
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    # Set the language token
    processor.tokenizer.set_prefix_tokens(language=language, task="transcribe")

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    if use_peft:
        if peft_config is None:
            peft_config = {
                "task_type": TaskType.SEQ_2_SEQ_LM,
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
            }

        # target_modules = []
        # for id, (name, param) in enumerate(model.named_modules()):
        #     if 'model.decoder' in name and ('q_proj' in name or 'v_proj' in name):
        #         target_modules.append(name)
        target_modules = [
            "q_proj",
            "v_proj",
        ]  # ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
        peft_config.update({"target_modules": target_modules})

        lora_config = LoraConfig(**peft_config)
        model = prepare_model_for_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor
