"""
Model Class
Load GPT-Neox model to generate text.
A function is given to finetune the model from a given checkpoint, save checkpoint, load checkpoint.
"""

import torch
from transformers import GPTJForCausalLM,AutoTokenizer
import os


class Model:
    def __init__(self):
        self.model = GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)

        # If available use the checkpoint from finetuned model
        if os.path.exists('checkpoint.pt'):
            self.load_checkpoint('checkpoint.pt')

    def predict(self, text, length=100):
        input_ids = torch.tensor(self.tokenizer.encode(
            text)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

        return self.tokenizer.decode(logits[0].argmax(-1).tolist())

    def finetune(self, text, length=100):
        input_ids = torch.tensor(self.tokenizer.encode(
            text)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

        return self.tokenizer.decode(logits[0].argmax(-1).tolist())

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
