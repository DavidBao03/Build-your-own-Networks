import torch
from sentencepiece import SentencePieceProcessor
from pathlib import Path
import json
import time

from llama import Transformer, ModelArgs

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args

    @staticmethod
    def build_model(checkpoint_path: str, tokenizer_path: str, load_model: bool, 
                    max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoint_path).glob("*.pth"))
            assert len(checkpoints) > 0, f"No checkpoints found in {checkpoint_path}"
            chk_path = checkpoints[0]
            print(f"Loading model from {chk_path}")
            checkpoint = torch.load(chk_path, map_location="cpu", weights_only=False)
            print(f"Lodaing model in {time.time() - prev_time:.2f} seconds")
            prev_time = time.time()
        
        with open(Path(checkpoint_path) / "params.json", "r") as f:
            params = json.loads(f.read())
        model_args = ModelArgs(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.Load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfStorage)
        else:
            torch.set_default_dtype(torch.bfloat16)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loading model weights in {time.time() - prev_time:.2f} seconds")


        return LLaMA(model, tokenizer, model_args)

if __name__ == "__main__":
    llama = LLaMA.build_model("./llama-2-7b/", "./llama-2-7b/tokenizer.model", True, 256, 1, "cpu")        
