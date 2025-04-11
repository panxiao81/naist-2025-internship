from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from peft import get_peft_model, LoraConfig, TaskType

model_name = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# lora_config = LoraConfig(
#     r=8,  # Rank of the low-rank matrices
#     lora_alpha=32,  # Scaling factor for LoRA matrices
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",
#                       "lm_head", "embed_tokens",],  # Attention layers to modify
#     task_type=TaskType.CAUSAL_LM  # For language modeling tasks
# )

# model = get_peft_model(model, lora_config)

# dataset = datasets.load_dataset