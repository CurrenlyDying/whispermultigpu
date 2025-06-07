# main.py
import argparse
import logging
from pathlib import Path
from datetime import datetime
from utils.logging_config import setup_logging
from core.model_handler import ModelHandler
from core.audio_converter import AudioConverter
from config.settings import CONFIG
from utils.file_handlers import save_results

logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio processing with Whisper model")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"],
                        default="auto", help="Device to use for processing")
    parser.add_argument("--language", default=None,
                        help="Language of the audio. Omit for auto-detection and multi-language support.")
    parser.add_argument("--model", default=CONFIG['processing'].default_model,
                        help="Model name or path")
    # MODIFICATION: Added --prompt argument
    parser.add_argument("--prompt", type=str, default=None,
                        help="Initial prompt to guide the model's transcription.")
    parser.add_argument("--input", required=True, type=Path,
                        help="Input audio file path")
    parser.add_argument("--task", choices=["transcribe", "translate"],
                        required=True, help="Task to perform")
    parser.add_argument("--output", type=Path,
                        help="Output file path for the result")
    return parser.parse_args()

def main() -> None:
    setup_logging()
    args = parse_arguments()

    try:
        start_time = datetime.now()
        logger.info(f"Starting processing at {start_time}")

        # Initialize model resources
        resources = ModelHandler.initialize(args.model, args.device)
        logger.info(f"Model initialized on {resources.device}")

        # Process audio file
        input_file = AudioConverter.ensure_compatible_audio(args.input)
        logger.info(f"Processing file: {input_file}")
        
        # Conditionally build generate_kwargs
        generate_kwargs = {
            "task": args.task
        }
        if args.language:
            logger.info(f"Language specified: {args.language}")
            generate_kwargs["language"] = args.language
        else:
            logger.info("No language specified. Using auto-detection.")
        
        # MODIFICATION: Add prompt to generate_kwargs if provided
        if args.prompt:
            logger.info(f"Using initial prompt: {args.prompt}")
            generate_kwargs["prompt"] = args.prompt


        # Process audio and generate output
        # MODIFICATION: Changed return_timestamps to 'word' for higher accuracy
        logger.info("Processing with word-level timestamps for improved accuracy.")
        result = resources.pipeline(
            str(input_file),
            return_timestamps='word',
            generate_kwargs=generate_kwargs
        )

        # Save results in both formats
        vtt_path, text_path = save_results(
            result,
            input_file,
            args.output
        )

        processing_time = datetime.now() - start_time
        logger.info(f"Processing completed in {processing_time}")
        logger.info(f"Results saved to:")
        logger.info(f"  VTT: {vtt_path}")
        logger.info(f"  Text: {text_path}")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
