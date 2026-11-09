"""CPU inference for the two pinned ONNX models, without torch."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def _session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _feed(session, encodings):
    feed = {
        "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
    }
    if any(item.name == "token_type_ids" for item in session.get_inputs()):
        feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
    return feed


class Encoder:
    """Sentence embeddings as raw vectors; callers choose whether to normalize."""

    def __init__(self, directory, threads=6, max_length=256, batch_size=32):
        directory = Path(directory)
        self.session = _session(directory / "model_quantized.onnx", threads)
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length)
        self.tokenizer.enable_padding()
        self.batch_size = batch_size

    def encode(self, texts, pooling):
        order = np.argsort([len(text) for text in texts], kind="stable")
        output = [None] * len(texts)
        for start in range(0, len(texts), self.batch_size):
            batch = order[start : start + self.batch_size]
            encodings = self.tokenizer.encode_batch([texts[i] for i in batch])
            feed = _feed(self.session, encodings)
            hidden = self.session.run(None, feed)[0]
            if pooling == "cls":
                vectors = hidden[:, 0]
            else:
                mask = feed["attention_mask"][..., None]
                vectors = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
            for index, vector in zip(batch, vectors):
                output[index] = vector
        return np.stack(output)


class EntailmentModel:
    """P(entailment) for (premise, hypothesis) pairs."""

    def __init__(self, directory, threads=6, max_length=256, batch_size=32):
        directory = Path(directory)
        self.session = _session(directory / "model_quantized.onnx", threads)
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length)
        self.tokenizer.enable_padding()
        self.batch_size = batch_size
        labels = json.loads((directory / "config.json").read_text(encoding="utf-8"))["id2label"]
        self.entailment = next(int(i) for i, name in labels.items() if name.lower() == "entailment")

    def __call__(self, pairs):
        order = np.argsort([len(p) + len(h) for p, h in pairs], kind="stable")
        output = np.zeros(len(pairs))
        for start in range(0, len(pairs), self.batch_size):
            batch = order[start : start + self.batch_size]
            encodings = self.tokenizer.encode_batch([pairs[i] for i in batch])
            logits = self.session.run(None, _feed(self.session, encodings))[0]
            exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
            output[batch] = (exponent / exponent.sum(axis=1, keepdims=True))[:, self.entailment]
        return output
