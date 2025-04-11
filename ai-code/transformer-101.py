from transformers import BertModel, BertTokenizer
import torch

model = BertModel.from_pretrained("bert-base-cased", device_map='xpu:0')
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sequence = "I've been waiting for a HuggingFace course my whole life."
tokens = tokenizer.tokenize(sequence)

ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor([ids]).to('xpu:0')

outputs = model(input_ids)

print(outputs.logits)