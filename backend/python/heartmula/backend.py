#!/usr/bin/env python3
"""
LocalAI gRPC backend for HeartMuLa music generation.

Maps the TTS API to HeartMuLa:
  - request.text  -> lyrics
  - request.voice -> tags (comma-separated, e.g. "piano,happy,wedding")
  - Options can override: temperature, topk, cfg_scale, max_audio_length_ms
"""
from concurrent import futures
import time
import argparse
import signal
import sys
import os
import tempfile

import backend_pb2
import backend_pb2_grpc
import grpc
import torch
import gc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_WORKERS = int(os.environ.get("PYTHON_GRPC_MAX_WORKERS", "1"))


def quantize_model_int8(model, device):
    """Apply int8 weight-only quantization via torchao to reduce VRAM."""
    try:
        from torchao.quantization import quantize_, int8_weight_only
        print("Applying int8 weight-only quantization to backbone...", file=sys.stderr)
        quantize_(model.backbone, int8_weight_only())
        print("Applying int8 weight-only quantization to decoder...", file=sys.stderr)
        quantize_(model.decoder, int8_weight_only())
        gc.collect()
        torch.cuda.empty_cache()
        mem_gb = torch.cuda.memory_allocated(device) / 1024**3
        print(f"Quantization complete. VRAM on {device}: {mem_gb:.2f} GB", file=sys.stderr)
    except Exception as e:
        print(f"Warning: int8 quantization failed ({e}), running in bf16", file=sys.stderr)


def force_unload(pipe, component, device):
    """Aggressively free a pipeline component's VRAM.

    The stock _unload() doesn't reset KV caches, doesn't specify device for
    empty_cache, and doesn't break circular refs from quantized tensors.
    """
    obj = getattr(pipe, f"_{component}", None)
    if obj is None:
        return

    label = component.upper()
    mem_before = torch.cuda.memory_allocated(device) / 1024**3
    print(f"Unloading {label} (VRAM before: {mem_before:.2f} GB on {device})", file=sys.stderr)

    # 1. Reset KV caches (they hold pre-allocated GPU tensors)
    if hasattr(obj, "reset_caches"):
        try:
            obj.reset_caches()
        except Exception:
            pass

    # 2. Move all parameters/buffers to CPU to release CUDA tensors
    try:
        obj.cpu()
    except Exception:
        pass

    # 3. Clear all parameter references
    setattr(pipe, f"_{component}", None)

    # 4. Force garbage collection and CUDA cache flush
    del obj
    gc.collect()
    torch.cuda.empty_cache()

    mem_after = torch.cuda.memory_allocated(device) / 1024**3
    print(f"Unloaded {label} (VRAM after: {mem_after:.2f} GB on {device}, freed {mem_before - mem_after:.2f} GB)", file=sys.stderr)


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


class BackendServicer(backend_pb2_grpc.BackendServicer):
    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", "utf-8"))

    def LoadModel(self, request, context):
        try:
            from huggingface_hub import snapshot_download

            # Determine checkpoint directory
            model_dir = request.ModelFile if request.ModelFile else "/models/heartmula-ckpt"

            # Parse options
            self.options = {}
            for opt in request.Options:
                if ":" not in opt:
                    continue
                key, value = opt.split(":", 1)
                if is_float(value):
                    value = float(value)
                elif is_int(value):
                    value = int(value)
                elif value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                self.options[key] = value

            # Download checkpoints if not present
            ckpt_dir = self.options.get("ckpt_dir", model_dir)
            if not os.path.isdir(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)

            gen_config = os.path.join(ckpt_dir, "gen_config.json")
            if not os.path.exists(gen_config):
                print(f"Downloading HeartMuLaGen base files to {ckpt_dir}...", file=sys.stderr)
                snapshot_download("HeartMuLa/HeartMuLaGen", local_dir=ckpt_dir)

            mula_dir = os.path.join(ckpt_dir, "HeartMuLa-oss-3B")
            if not os.path.isdir(mula_dir) or not os.listdir(mula_dir):
                print(f"Downloading HeartMuLa-oss-3B weights to {mula_dir}...", file=sys.stderr)
                snapshot_download("HeartMuLa/HeartMuLa-oss-3B-happy-new-year", local_dir=mula_dir)

            codec_dir = os.path.join(ckpt_dir, "HeartCodec-oss")
            if not os.path.isdir(codec_dir) or not os.listdir(codec_dir):
                print(f"Downloading HeartCodec-oss to {codec_dir}...", file=sys.stderr)
                snapshot_download("HeartMuLa/HeartCodec-oss-20260123", local_dir=codec_dir)

            # Device configuration
            mula_device_str = str(self.options.get("mula_device", "cuda:1"))
            codec_device_str = str(self.options.get("codec_device", "cuda:0"))

            # Dtype configuration
            mula_dtype_str = str(self.options.get("mula_dtype", "bfloat16"))
            codec_dtype_str = str(self.options.get("codec_dtype", "float32"))
            dtype_map = {
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16,
                "float32": torch.float32, "fp32": torch.float32,
            }
            mula_dtype = dtype_map.get(mula_dtype_str, torch.bfloat16)
            codec_dtype = dtype_map.get(codec_dtype_str, torch.float32)

            mula_device = torch.device(mula_device_str)
            codec_device = torch.device(codec_device_str)

            print(f"Loading HeartMuLa from {ckpt_dir}", file=sys.stderr)
            print(f"  mula_device={mula_device}, codec_device={codec_device}", file=sys.stderr)
            print(f"  mula_dtype={mula_dtype}, codec_dtype={codec_dtype}", file=sys.stderr)

            from heartlib import HeartMuLaGenPipeline

            # Construct with mula_device for both so lazy_load stays True,
            # then monkey-patch codec_device to point to a different GPU.
            self.pipe = HeartMuLaGenPipeline.from_pretrained(
                ckpt_dir,
                device=mula_device,
                dtype={
                    "mula": mula_dtype,
                    "codec": codec_dtype,
                },
                version="3B",
                lazy_load=True,
            )

            # Patch codec to use a different device (pipeline disables
            # lazy_load when devices differ, so we set it after construction)
            self.pipe.codec_device = codec_device
            self.mula_device = mula_device
            self.codec_device = codec_device
            self.quantize = self.options.get("quantize", True)
            self.quantized = False

            # Monkey-patch _unload to aggressively free VRAM
            pipe = self.pipe
            md = mula_device
            cd = codec_device
            servicer = self
            def patched_unload():
                if not pipe.lazy_load:
                    return
                force_unload(pipe, "mula", md)
                force_unload(pipe, "codec", cd)
                # Reset quantized flag so next load re-quantizes the fresh model
                servicer.quantized = False
            self.pipe._unload = patched_unload

            print(f"  lazy_load={self.pipe.lazy_load} (patched codec_device={codec_device})", file=sys.stderr)
            print(f"  quantize={self.quantize}", file=sys.stderr)

            # Default generation params
            self.default_temperature = float(self.options.get("temperature", 1.0))
            self.default_topk = int(self.options.get("topk", 50))
            self.default_cfg_scale = float(self.options.get("cfg_scale", 1.5))
            self.default_max_ms = int(self.options.get("max_audio_length_ms", 240000))
            self.default_tags = str(self.options.get("default_tags", ""))

            print("HeartMuLa loaded successfully", file=sys.stderr)
            return backend_pb2.Result(message="Model loaded successfully", success=True)

        except Exception as err:
            print(f"Error loading HeartMuLa: {err}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return backend_pb2.Result(success=False, message=f"Unexpected {err=}, {type(err)=}")

    def TTS(self, request, context):
        lyrics_path = None
        tags_path = None
        try:
            lyrics = request.text
            # Tags come from the voice field, falling back to default_tags
            tags = request.voice if request.voice else self.default_tags

            # Write lyrics and tags to temp files (HeartMuLa reads from file paths)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as lf:
                lf.write(lyrics)
                lyrics_path = lf.name

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
                tf.write(tags)
                tags_path = tf.name

            print(f"Generating music: lyrics={len(lyrics)} chars, tags='{tags}'", file=sys.stderr)

            # Trigger lazy load + quantize before inference
            if self.quantize and not self.quantized:
                _ = self.pipe.mula  # trigger lazy load
                quantize_model_int8(self.pipe._mula, self.mula_device)
                self.quantized = True

            with torch.no_grad():
                self.pipe(
                    {
                        "lyrics": lyrics_path,
                        "tags": tags_path,
                    },
                    max_audio_length_ms=self.default_max_ms,
                    save_path=request.dst,
                    topk=self.default_topk,
                    temperature=self.default_temperature,
                    cfg_scale=self.default_cfg_scale,
                )

            # Clean up temp files
            os.unlink(lyrics_path)
            os.unlink(tags_path)
            lyrics_path = None
            tags_path = None

            print(f"Music generated -> {request.dst}", file=sys.stderr)
            return backend_pb2.Result(success=True)

        except Exception as err:
            print(f"Error generating music: {err}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Clean up on error
            for p in [lyrics_path, tags_path]:
                if p:
                    try:
                        os.unlink(p)
                    except Exception:
                        pass
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
    print("HeartMuLa server started. Listening on: " + address, file=sys.stderr)

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
    parser = argparse.ArgumentParser(description="Run the HeartMuLa gRPC server.")
    parser.add_argument("--addr", default="localhost:50051", help="The address to bind the server to.")
    args = parser.parse_args()
    serve(args.addr)
