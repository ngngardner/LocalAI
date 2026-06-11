#!/usr/bin/env python3
"""
LocalAI gRPC backend for MisoTTS (MisoLabsAI/MisoTTS).

MisoTTS 8B is a text-to-dialogue RVQ transformer (Sesame CSM architecture):
a llama-8B backbone + llama-300M audio decoder over the Mimi codec.

Maps the TTS API:
  - request.text  -> text to speak
  - request.voice -> speaker id (integer, default 0)

VRAM strategy for consumer GPUs (e.g. 12GB RTX 3060): the model is loaded
on CPU in bf16, the backbone/decoder are quantized with torchao weight-only
quantization (int8 default, int4 optional), and only then moved to the GPU.
Embeddings/heads stay bf16; KV caches are created bf16 because torchao's
quantized tensors report the original float dtype.

Options (model YAML):
  - gpu_device: restrict to one physical GPU (e.g. "1" or "cuda:1").
    Sets CUDA_VISIBLE_DEVICES before torch import (cuInit caches it).
  - quantize: "int8" (default) | "int4" | "none"
  - model_repo: HF repo or local path (default MisoLabs/MisoTTS)
  - tokenizer_repo: text tokenizer repo (default unsloth/Llama-3.2-1B,
    a public mirror of the gated meta-llama/Llama-3.2-1B tokenizer)
  - hf_home: HF cache dir (default /models/miso-hf)
  - temperature, topk, max_audio_length_ms, default_speaker
  - chunk_chars: max characters per generation chunk (default 300)
  - context_segments: prior segments fed as voice-consistency context (default 1)
"""
from concurrent import futures
import time
import argparse
import signal
import sys
import os
import gc

import grpc

import backend_pb2
import backend_pb2_grpc

# torch and the miso modules are imported lazily in LoadModel so that
# CUDA_VISIBLE_DEVICES / HF_HOME can be set first from model config options.
torch = None

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_WORKERS = int(os.environ.get("PYTHON_GRPC_MAX_WORKERS", "1"))


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def parse_options(raw_options):
    """Parse key:value option strings into a dict."""
    options = {}
    for opt in raw_options:
        if ":" not in opt:
            continue
        key, value = opt.split(":", 1)
        if is_float(value):
            value = float(value)
        elif is_int(value):
            value = int(value)
        elif value.lower() in ["true", "false"]:
            value = value.lower() == "true"
        options[key] = value
    return options


def coerce_param_value(value):
    """Coerce a TTSRequest.params value (string on the wire) to float/int/bool."""
    if not isinstance(value, str):
        return value
    if is_float(value):
        return float(value)
    if is_int(value):
        return int(value)
    if value.lower() in ["true", "false"]:
        return value.lower() == "true"
    return value


def split_text_into_chunks(text, max_length=300):
    """Split text at sentence boundaries (falling back to word boundaries)
    into chunks of at most max_length characters."""
    import re

    if not text or len(text) <= max_length:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for sentence in sentences:
        # A single sentence longer than max_length gets word-split below.
        if len(sentence) > max_length:
            if current:
                chunks.append(current)
                current = ""
            words = sentence.split()
            piece = ""
            for word in words:
                if len(piece) + len(word) + 1 <= max_length:
                    piece = f"{piece} {word}".strip()
                else:
                    if piece:
                        chunks.append(piece)
                    piece = word
            if piece:
                current = piece
            continue
        if len(current) + len(sentence) + 1 <= max_length:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


class BackendServicer(backend_pb2_grpc.BackendServicer):
    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", "utf-8"))

    def LoadModel(self, request, context):
        global torch
        try:
            # 1. Parse options BEFORE importing torch / hf libs.
            self.options = parse_options(request.Options)

            if "gpu_device" in self.options:
                gpu_id = str(self.options.pop("gpu_device"))
                if gpu_id.startswith("cuda:"):
                    gpu_id = gpu_id.split(":")[1]
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
                print(f"Set CUDA_VISIBLE_DEVICES={gpu_id}", file=sys.stderr)

            hf_home = str(self.options.pop("hf_home", "/models/miso-hf"))
            os.environ.setdefault("HF_HOME", hf_home)
            os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
            os.environ.setdefault("NO_TORCH_COMPILE", "1")

            # 2. Import torch (cuInit reads CUDA_VISIBLE_DEVICES here).
            import torch as _torch
            torch = _torch

            import generator as miso_generator
            from models import MISO_TTS_8B_CONFIG

            # 3. The stock tokenizer repo (meta-llama/Llama-3.2-1B) is gated on
            # HF; default to a public mirror of the same tokenizer.
            tokenizer_repo = str(self.options.pop("tokenizer_repo", "unsloth/Llama-3.2-1B"))

            def load_tokenizer():
                from tokenizers.processors import TemplateProcessing
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
                bos = tokenizer.bos_token
                eos = tokenizer.eos_token
                tokenizer._tokenizer.post_processor = TemplateProcessing(
                    single=f"{bos}:0 $A:0 {eos}:0",
                    pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
                    special_tokens=[
                        (f"{bos}", tokenizer.bos_token_id),
                        (f"{eos}", tokenizer.eos_token_id),
                    ],
                )
                return tokenizer

            miso_generator.load_llama3_tokenizer = load_tokenizer

            # 4. Resolve model source and target device.
            source = request.ModelFile if request.ModelFile else None
            if not source or not os.path.exists(source):
                source = str(self.options.pop("model_repo", "MisoLabs/MisoTTS"))

            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            if device == "cpu" and request.CUDA:
                return backend_pb2.Result(success=False, message="CUDA is not available")

            quantize_mode = str(self.options.pop("quantize", "int8")).lower()

            # 5. Load on CPU in bf16 (8B bf16 = ~17GB, too large for a 12GB
            # card), quantize there, then move the shrunken model to the GPU.
            # The published checkpoint is fp32 (~32GB); constructing the empty
            # Model under a bf16 default dtype halves the anonymous-RAM peak
            # (the safetensors state dict itself is mmap-backed, and
            # load_state_dict casts fp32->bf16 as it copies).
            print(f"Loading MisoTTS from {source} on CPU (bf16)...", file=sys.stderr)
            torch.set_default_dtype(torch.bfloat16)
            try:
                model = miso_generator._load_model(
                    source, MISO_TTS_8B_CONFIG, device="cpu", dtype=torch.bfloat16
                )
            finally:
                torch.set_default_dtype(torch.float32)

            if quantize_mode in ("int8", "int4", "true"):
                from torchao.quantization import quantize_, int8_weight_only, int4_weight_only

                qcfg = int4_weight_only() if quantize_mode == "int4" else int8_weight_only()
                label = "int4" if quantize_mode == "int4" else "int8"
                print(f"Applying {label} weight-only quantization to backbone...", file=sys.stderr)
                quantize_(model.backbone, qcfg)
                print(f"Applying {label} weight-only quantization to decoder...", file=sys.stderr)
                quantize_(model.decoder, qcfg)
                gc.collect()

            if device != "cpu":
                print(f"Moving model to {device}...", file=sys.stderr)
                model.to(device)
                gc.collect()
                torch.cuda.empty_cache()

            # Generator() sets up KV caches, Mimi codec and the watermarker on
            # the model's device.
            print("Initializing generator (Mimi codec + watermarker)...", file=sys.stderr)
            self.generator = miso_generator.Generator(model)
            self.Segment = miso_generator.Segment

            if device != "cpu":
                mem_gb = torch.cuda.memory_allocated() / 1024**3
                print(f"MisoTTS loaded. VRAM allocated: {mem_gb:.2f} GB", file=sys.stderr)

            # 6. Generation defaults.
            self.default_temperature = float(self.options.get("temperature", 0.9))
            self.default_topk = int(self.options.get("topk", 50))
            self.default_max_ms = float(self.options.get("max_audio_length_ms", 60_000))
            self.default_speaker = int(self.options.get("default_speaker", 0))
            self.chunk_chars = int(self.options.get("chunk_chars", 300))
            self.context_segments = int(self.options.get("context_segments", 1))

            return backend_pb2.Result(message="Model loaded successfully", success=True)

        except Exception as err:
            print(f"Error loading MisoTTS: {err}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
            return backend_pb2.Result(success=False, message=f"Unexpected {err=}, {type(err)=}")

    def TTS(self, request, context):
        try:
            import torchaudio

            params = {
                "temperature": self.default_temperature,
                "topk": self.default_topk,
                "max_audio_length_ms": self.default_max_ms,
                "speaker": self.default_speaker,
            }
            # Per-request params (TTSRequest.params) override YAML defaults.
            if hasattr(request, "params") and request.params:
                for key, value in request.params.items():
                    params[key] = coerce_param_value(value)

            # voice field carries the speaker id, e.g. voice="0"
            if request.voice and is_int(request.voice):
                params["speaker"] = int(request.voice)

            speaker = int(params["speaker"])
            chunks = split_text_into_chunks(request.text, max_length=self.chunk_chars)
            print(
                f"Generating speech: {len(request.text)} chars in {len(chunks)} chunk(s), "
                f"speaker={speaker}",
                file=sys.stderr,
            )

            # Generate chunk-by-chunk, feeding previous segments back as
            # context — without it each chunk would get a different voice.
            segments = []
            audio_parts = []
            for i, chunk in enumerate(chunks):
                start = time.time()
                ctx = segments[-self.context_segments:] if self.context_segments > 0 else []
                audio = self.generator.generate(
                    text=chunk,
                    speaker=speaker,
                    context=ctx,
                    max_audio_length_ms=float(params["max_audio_length_ms"]),
                    temperature=float(params["temperature"]),
                    topk=int(params["topk"]),
                )
                elapsed = time.time() - start
                secs = audio.shape[-1] / self.generator.sample_rate
                print(
                    f"  chunk {i + 1}/{len(chunks)}: {secs:.1f}s audio in {elapsed:.1f}s",
                    file=sys.stderr,
                )
                segments.append(self.Segment(speaker=speaker, text=chunk, audio=audio))
                audio_parts.append(audio)

            merged = torch.cat(audio_parts, dim=-1) if len(audio_parts) > 1 else audio_parts[0]
            torchaudio.save(
                request.dst,
                merged.unsqueeze(0).to("cpu", torch.float32),
                self.generator.sample_rate,
            )

            print(f"Speech generated -> {request.dst}", file=sys.stderr)
            return backend_pb2.Result(success=True)

        except Exception as err:
            print(f"Error generating speech: {err}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
            return backend_pb2.Result(success=False, message=f"Unexpected {err=}, {type(err)=}")


def serve(address):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_message_length", 100 * 1024 * 1024),
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )
    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)
    server.add_insecure_port(address)
    server.start()
    print("MisoTTS server started. Listening on: " + address, file=sys.stderr)

    def signal_handler(sig, frame):
        print("Received termination signal. Shutting down...")
        server.stop(0)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MisoTTS gRPC server.")
    parser.add_argument("--addr", default="localhost:50051", help="The address to bind the server to.")
    args = parser.parse_args()
    serve(args.addr)
