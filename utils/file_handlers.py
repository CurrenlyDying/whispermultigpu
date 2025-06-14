# utils/file_handlers.py
from pathlib import Path
from typing import Dict, List, Union, Optional
import logging

logger = logging.getLogger(__name__)

class OutputHandler:
    @staticmethod
    def write_vtt(
            chunks: List[Dict[str, Union[str, List[float]]]],
            output_path: Path
    ) -> None:
        """
        Write transcription/translation results in VTT format with timestamps.

        Args:
            chunks: List of dictionaries containing text and timestamp information
            output_path: Path to the output VTT file
        """
        try:
            with output_path.open('w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                for i, chunk in enumerate(chunks, start=1):
                    # MODIFICATION: Check for missing timestamps to prevent crashes.
                    # This can happen if the audio cuts off abruptly.
                    if chunk.get('timestamp') is None or chunk['timestamp'][1] is None:
                        logger.warning(
                            f"Skipping chunk with missing timestamp: \"{chunk.get('text', '').strip()}\""
                        )
                        continue

                    start_time = OutputHandler._format_timestamp(chunk['timestamp'][0])
                    end_time = OutputHandler._format_timestamp(chunk['timestamp'][1])
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{chunk['text'].strip()}\n\n")
            logger.info(f"VTT file written to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write VTT file: {str(e)}")
            raise

    @staticmethod
    def write_text(
            text: str,
            output_path: Path
    ) -> None:
        """
        Write plain text transcription/translation without timestamps.

        Args:
            text: The transcribed/translated text
            output_path: Path to the output text file
        """
        try:
            with output_path.open('w', encoding='utf-8') as f:
                f.write(text.strip())
            logger.info(f"Text file written to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write text file: {str(e)}")
            raise

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """
        Format timestamp in HH:MM:SS.ms format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string
        """
        # MODIFICATION: Handle cases where seconds might be None, although the primary check is in write_vtt.
        if seconds is None:
            return "00:00:00.000"
            
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)
        
        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        secs = whole_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

    @staticmethod
    def get_output_paths(
            input_path: Path,
            output_path: Optional[Path] = None
    ) -> tuple[Path, Path]:
        """
        Generate output file paths for both VTT and text files.

        Args:
            input_path: Original input file path
            output_path: Optional specified output path

        Returns:
            Tuple of (vtt_path, text_path)
        """
        if output_path:
            # If output path is specified, use it as a base
            base_path = output_path.parent / output_path.stem
        else:
            # Otherwise use input path as a base
            base_path = input_path.parent / input_path.stem

        vtt_path = base_path.with_suffix('.vtt')
        text_path = base_path.with_suffix('.txt')

        return vtt_path, text_path

def save_results(
        result: Dict[str, Union[str, List[Dict]]],
        input_path: Path,
        output_path: Optional[Path] = None
) -> tuple[Path, Path]:
    """
    Save processing results in both VTT and text formats.

    Args:
        result: Dictionary containing processing results
        input_path: Original input file path
        output_path: Optional specified output path

    Returns:
        Tuple of paths where results were saved (vtt_path, text_path)
    """
    vtt_path, text_path = OutputHandler.get_output_paths(input_path, output_path)

    # Save VTT format with timestamps
    if 'chunks' in result and result['chunks'] is not None:
        OutputHandler.write_vtt(result['chunks'], vtt_path)
    else:
        logger.warning("No 'chunks' found in result, skipping VTT file creation.")


    # Save plain text format
    OutputHandler.write_text(result['text'], text_path)

    return vtt_path, text_path
