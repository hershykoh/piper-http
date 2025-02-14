#!/usr/bin/env python3

import numpy as np
np.int = np.int32
np.float = np.float64
np.bool = np.bool_
import argparse
import io
import logging
import wave
from pathlib import Path
from typing import Any, Dict

from flask import Flask, request, send_file

from . import PiperVoice
from .download import ensure_voice_exists, find_voice, get_voices
import nltk
# Download necessary NLTK resources
for dependency in ("brown", "names", "wordnet", "averaged_perceptron_tagger", "universal_tagset","averaged_perceptron_tagger_eng"):
    nltk.download(dependency)
from normalise import tokenize_basic, normalise



_LOGGER = logging.getLogger()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host")
    parser.add_argument("--port", type=int, default=5000, help="HTTP server port")
    #
    parser.add_argument("-m", "--model", required=True, help="Path to Onnx model file")
    parser.add_argument("-c", "--config", help="Path to model config file")
    #
    parser.add_argument("-s", "--speaker", type=int, help="Id of speaker (default: 0)")
    parser.add_argument(
        "--length-scale", "--length_scale", type=float, help="Phoneme length"
    )
    parser.add_argument(
        "--noise-scale", "--noise_scale", type=float, help="Generator noise"
    )
    parser.add_argument(
        "--noise-w", "--noise_w", type=float, help="Phoneme width noise"
    )
    #
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    #
    parser.add_argument(
        "--sentence-silence",
        "--sentence_silence",
        type=float,
        default=0.0,
        help="Seconds of silence after each sentence",
    )
    #
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        action="append",
        default=[str(Path.cwd())],
        help="Data directory to check for downloaded models (default: current directory)",
    )
    parser.add_argument(
        "--download-dir",
        "--download_dir",
        help="Directory to download voices into (default: first data dir)",
    )
    #
    parser.add_argument(
        "--update-voices",
        action="store_true",
        help="Download latest voices.json during startup",
    )
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    if not args.download_dir:
        # Download to first data directory by default
        args.download_dir = args.data_dir[0]

    # Download voice if file doesn't exist
    model_path = Path(args.model)
    if not model_path.exists():
        # Load voice info
        voices_info = get_voices(args.download_dir, update_voices=args.update_voices)

        # Resolve aliases for backwards compatibility with old voice names
        aliases_info: Dict[str, Any] = {}
        for voice_info in voices_info.values():
            for voice_alias in voice_info.get("aliases", []):
                aliases_info[voice_alias] = {"_is_alias": True, **voice_info}

        voices_info.update(aliases_info)
        ensure_voice_exists(args.model, args.data_dir, args.download_dir, voices_info)
        args.model, args.config = find_voice(args.model, args.data_dir)

    # Load voice
    voice = PiperVoice.load(args.model, config_path=args.config, use_cuda=args.cuda)
    synthesize_args = {
        "speaker_id": args.speaker,
        "length_scale": args.length_scale,
        "noise_scale": args.noise_scale,
        "noise_w": args.noise_w,
        "sentence_silence": args.sentence_silence,
    }

    # Create web server
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def app_synthesize() -> bytes:
        if request.method == "POST":
            text = request.data.decode("utf-8")
        else:
            text = request.args.get("text", "")

        text = text.strip()
        if not text:
            raise ValueError("No text provided")

        out = normalise(text, tokenizer=tokenize_basic, verbose=False)
        text = ' '.join(out)
        _LOGGER.debug("Synthesizing text: %s", text)
        # Desired settings: 16-bit, 8kHz (or 16kHz), Mono
        sample_width = 2  # 16 bits = 2 bytes
        sample_rate = 16000  # 8kHz or 16kHz - change this if needed (e.g., 8000 for 8kHz)
        num_channels = 1  # Mono

        # Create the audio data in memory
        wav_io = io.BytesIO()

        with wave.open(wav_io, "wb") as wav_file:
            # Set the parameters for the WAV file (16-bit, 16kHz Mono)
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)

            # Synthesize speech and write audio frames into the WAV file
            # `voice.synthesize` should generate the audio in the expected format.
            voice.synthesize(text, wav_file, **synthesize_args)

        # Ensure the file pointer is at the beginning of the BytesIO object before sending
        wav_io.seek(0)

        return send_file(
            wav_io,
            as_attachment=True,
            download_name="synthesized_audio.wav",  # Name of the file when downloading
            mimetype="audio/wav"
        )

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
