#!/usr/bin/env python3
"""
LocalAI gRPC backend for Chatterbox TTS.

GPU selection: set gpu_device option in model YAML (e.g. "gpu_device:cuda:1")
to restrict which physical GPU is used. This sets CUDA_VISIBLE_DEVICES before
torch is imported, since the CUDA driver caches device visibility at cuInit() time.
"""
from concurrent import futures
import time
import argparse
import signal
import sys
import os
import grpc
import tempfile

import backend_pb2
import backend_pb2_grpc

# torch, torchaudio, chatterbox are imported lazily in LoadModel
# so that CUDA_VISIBLE_DEVICES can be set first from model config options.
torch = None
ta = None


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


def split_text_at_word_boundary(text, max_length=250):
    if not text or len(text) <= max_length:
        return [text]
    chunks = []
    words = text.split()
    current_chunk = ""
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_length:
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = word
            else:
                chunks.append(word)
                current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def merge_audio_files(audio_files, output_path, sample_rate):
    if not audio_files:
        return
    if len(audio_files) == 1:
        import shutil
        shutil.copy2(audio_files[0], output_path)
        return
    waveforms = []
    for audio_file in audio_files:
        waveform, sr = ta.load(audio_file)
        if sr != sample_rate:
            resampler = ta.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        waveforms.append(waveform)
    merged_waveform = torch.cat(waveforms, dim=1)
    ta.save(output_path, merged_waveform, sample_rate)
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            os.remove(audio_file)


_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_WORKERS = int(os.environ.get('PYTHON_GRPC_MAX_WORKERS', '1'))


class BackendServicer(backend_pb2_grpc.BackendServicer):
    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", 'utf-8'))

    def LoadModel(self, request, context):
        global torch, ta

        try:
            # 1. Parse options BEFORE importing torch.
            #    CUDA_VISIBLE_DEVICES must be set before cuInit() which happens at import time.
            self.options = parse_options(request.Options)

            if "gpu_device" in self.options:
                gpu_id = str(self.options.pop("gpu_device"))
                if gpu_id.startswith("cuda:"):
                    gpu_index = gpu_id.split(":")[1]
                else:
                    gpu_index = gpu_id
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
                print(f"Set CUDA_VISIBLE_DEVICES={gpu_index}", file=sys.stderr)

            # 2. Now import torch — cuInit() reads CUDA_VISIBLE_DEVICES here.
            if torch is None:
                import torch as _torch
                import torchaudio as _ta
                torch = _torch
                ta = _ta

            # 3. Determine device
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            if device == "cpu" and request.CUDA:
                return backend_pb2.Result(success=False, message="CUDA is not available")

            # 4. Audio prompt path
            self.AudioPath = None
            if os.path.isabs(request.AudioPath):
                self.AudioPath = request.AudioPath
            elif request.AudioPath and request.ModelFile != "" and not os.path.isabs(request.AudioPath):
                modelFileBase = os.path.dirname(request.ModelFile)
                self.AudioPath = os.path.join(modelFileBase, request.AudioPath)

            # 5. Load model
            print("Preparing models, please wait", file=sys.stderr)
            if "multilingual" in self.options:
                del self.options["multilingual"]
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self.model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            else:
                from chatterbox.tts import ChatterboxTTS
                self.model = ChatterboxTTS.from_pretrained(device=device)

            return backend_pb2.Result(message="Model loaded successfully", success=True)
        except Exception as err:
            return backend_pb2.Result(success=False, message=f"Unexpected {err=}, {type(err)=}")

    def TTS(self, request, context):
        try:
            kwargs = {}
            if "language" in self.options:
                kwargs["language_id"] = self.options["language"]
            if self.AudioPath is not None:
                kwargs["audio_prompt_path"] = self.AudioPath
            kwargs.update(self.options)

            if len(request.text) > 250:
                text_chunks = split_text_at_word_boundary(request.text, max_length=250)
                print(f"Splitting text into {len(text_chunks)} chunks", file=sys.stderr)
                temp_audio_files = []
                for chunk in text_chunks:
                    wav = self.model.generate(chunk, **kwargs)
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_file.close()
                    ta.save(temp_file.name, wav, self.model.sr)
                    temp_audio_files.append(temp_file.name)
                merge_audio_files(temp_audio_files, request.dst, self.model.sr)
            else:
                wav = self.model.generate(request.text, **kwargs)
                ta.save(request.dst, wav, self.model.sr)

        except Exception as err:
            return backend_pb2.Result(success=False, message=f"Unexpected {err=}, {type(err)=}")
        return backend_pb2.Result(success=True)


def serve(address):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ('grpc.max_message_length', 50 * 1024 * 1024),
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ])
    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)
    server.add_insecure_port(address)
    server.start()
    print("Server started. Listening on: " + address, file=sys.stderr)

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
    parser = argparse.ArgumentParser(description="Run the gRPC server.")
    parser.add_argument(
        "--addr", default="localhost:50051", help="The address to bind the server to."
    )
    args = parser.parse_args()
    serve(args.addr)
