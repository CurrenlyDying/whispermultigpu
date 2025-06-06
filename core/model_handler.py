# core/model_handler.py
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline as transformers_pipeline
)
import logging
from config.settings import CONFIG

logger = logging.getLogger(__name__)

@dataclass
class ModelResources:
    model: AutoModelForSpeechSeq2Seq
    processor: AutoProcessor
    pipeline: any  # Using 'any' as the pipeline type is complex
    device: str
    dtype: torch.dtype

class ModelHandler:
    @staticmethod
    def get_device(device_arg: str) -> str:
        """Determine the appropriate device for model execution."""
        if device_arg == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device_arg

    @staticmethod
    def get_torch_dtype(device: str) -> torch.dtype:
        """Determine appropriate torch dtype based on device."""
        return torch.float16 if device.startswith(("cuda", "mps")) else torch.float32

    @classmethod
    def create_pipeline(cls, model, processor, dtype, device):
        """Create the pipeline with proper configuration."""
        try:
            # When using device_map, we don't need to specify the device again in the pipeline
            return transformers_pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=dtype,
                chunk_length_s=CONFIG['processing'].chunk_length_s,
                batch_size=CONFIG['processing'].batch_size,
            )
        except Exception as e:
            logger.error(f"Failed to create pipeline: {str(e)}")
            raise RuntimeError(f"Pipeline creation failed: {str(e)}") from e

    @classmethod
    def initialize(cls, model_name: str, device_arg: str = "auto") -> ModelResources:
        """Initialize model resources with proper error handling and multi-GPU support."""
        try:
            # This logic remains useful for determining dtype
            device = cls.get_device(device_arg)
            dtype = cls.get_torch_dtype(device)

            logger.info(f"Loading model {model_name} with automatic device mapping.")

            # MODIFICATION: Added device_map="auto" to enable multi-GPU usage via Accelerate.
            # This automatically distributes the model across available GPUs.
            # Removed the explicit .to(device) call as it's now handled by device_map.
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                device_map="auto" # <-- This is the key change
            )

            processor = AutoProcessor.from_pretrained(model_name)

            pipe = cls.create_pipeline(model, processor, dtype, device)
            
            # The 'device' here is the primary device, but the model may be on multiple devices.
            # We can log the model's actual device map if needed.
            logger.info(f"Model successfully loaded. Device map: {model.hf_device_map}")

            return ModelResources(model, processor, pipe, device, dtype)

        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e