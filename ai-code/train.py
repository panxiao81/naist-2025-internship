import datasets

from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

max_seq_length = 2048
dtype=None
load_in_4bit = True

model_name = 'Qwen/Qwen2.5-Coder-7B-Instruct'
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

dataset = datasets.load_dataset('parquet', data_files='document-dataset.parquet')
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    return { "text": [example + tokenizer.eos_token for example in examples["text"]] }

dataset = dataset['train'].map(formatting_prompts_func, batched=True)

#for row in dataset[:5]["text"]:
#    print("-----------------")
#    print(row)

# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

#lora_config = LoraConfig(
#    r=8,  # Rank of the low-rank matrices
#    lora_alpha=32,  # Scaling factor for LoRA matrices
#    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
#                      "gate_proj", "up_proj", "down_proj",
#                      "lm_head", "embed_tokens",],  # Attention layers to modify
#    task_type=TaskType.CAUSAL_LM,  # For language modeling tasks,
#    lora_dropout=0.1,
#)

# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()


# tokenized_datasets = dataset.map(lambda b: tokenizer(b['text'], return_tensors='pt', truncation=True), batched=True)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    use_rslora=False,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", # Add LoRA to all of the attention matrices
                      "gate_proj", "up_proj", "down_proj", # Add LoRA to all of the FFN matrices
                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_dropout = 0, 
    bias = 'none',
    use_gradient_checkpointing = 'unsloth',
    random_state = 3407,
    loftq_config = None,
)

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt')

#train_dataloader = DataLoader(
#    tokenized_datasets['train'], shuffle=True, batch_size=8, collate_fn=data_collator
#)

#training_args = TrainingArguments(
#    output_dir='./results',
#    num_train_epochs=3,  # Set the number of epochs
#    per_device_train_batch_size=1,
#    logging_dir='./logs',
#    logging_steps=10,
#    save_steps=100,
#    warmup_steps=500,
#    weight_decay=0.01,
#)

#trainer = Trainer(
#    model=model,
#    args=training_args,
#    train_dataset=tokenized_datasets['train'],
#    eval_dataset=None,
#    data_collator=data_collator,
#    tokenizer=tokenizer
#)

import gc
gc.collect()
#import torch
#torch.cuda.empty_cache()
#trainer.train()

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    dataset_num_proc = 4,
    max_seq_length = max_seq_length,
    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        embedding_learning_rate = 2e-5,
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.1,
        logging_steps = 10,
        report_to = 'none',
        seed=3407,
        output_dir = 'result'
    ),
)

trainer_stats = trainer.train()

system_prompt = """
You are TerraformAI, an AI agent that builds and deploys Cloud Infrastructure written in Terraform HCL. Generate a description of the Terraform program you will define, followed by a single Terraform HCL program in response to each of my Instructions. Make sure the configuration is deployable. Create IAM roles as needed. If variables are used, make sure default values are supplied. Be sure to include a valid provider configuration within a valid region. Make sure there are no undeclared resources (e.g., as references) or variables, that is, all resources and variables needed in the configuration should be fully specified.
"""

cot_prompt = """
Here are a few examples:

Example prompt 1: Create an AWS RDS instance (with an instance class of db.t2.micro, and don't create a final snapshot before eventual deletion) with randomly generated id and password
Example output 1: Let's think step by step. First, let's reason about the resources needed: this would be an AWS RDS instance (aws_db_instance), and resources to generate a random id and password. Second, we fill in the attributes of each resource, starting with those explicitly and implicitly mentioned in the prompt, and followed by others: for example, for the aws_db_instance, we need to set the "instance_class" attribute to "db.t2.micro", and the "skip_final_snapshot" attribute to true. Finally, we connect the resources together, as needed: here "identifier" should be connected to the "random_id" resource, and "password" should be connected to the "random_password" resource
```hcl
resource "random_id" "suffix" {
  byte_length = 4
}

resource "random_password" "db" {
  length  = 16
  special = false
}

resource "aws_db_instance" "test" {
  identifier          = "metricbeat-test-${random_id.suffix.hex}"
  allocated_storage   = 20 // Gigabytes
  engine              = "mysql"
  instance_class      = "db.t2.micro"
  db_name                = "metricbeattest"
  username            = "foo"
  password            = random_password.db.result
  skip_final_snapshot = true // Required for cleanup
}
```

Example prompt 2: Create an 20GB MySQL instance on aws with randomly generated id and password
Example output 2: Let's think step by step. First, let's reason about the resources needed: this would be an AWS RDS instance (aws_db_instance), and resources to generate a random id and password. Second, we fill in the attributes of each resource, starting with those explicitly and implicitly mentioned in the prompt, and followed by others: for example, for the aws_db_instance, we need to set the "engine" attribute to "mysql". Finally, we connect the resources together, as needed: here "identifier" should be connected to the "random_id" resource, and "password" should be connected to the "random_password" resource
```hcl
resource "random_id" "suffix" {
  byte_length = 4
}

resource "random_password" "db" {
  length  = 16
  special = false
}

resource "aws_db_instance" "test" {
  identifier          = "metricbeat-test-${random_id.suffix.hex}"
  allocated_storage   = 20 // Gigabytes
  engine              = "mysql"
  instance_class      = "db.t2.micro"
  db_name                = "metricbeattest"
  username            = "foo"
  password            = random_password.db.result
  skip_final_snapshot = true // Required for cleanup
}
```

Example prompt 3: create a AWS EFS, and create a replica of an this created EFS file system using regional storage in us-west-2
Example output 3: Let's think step by step. First, let's reason about the resources needed: this would be an AWS EFS replication resource (aws_efs_replication_configuration), and  the AWS EFS resource itself. Second, we fill in the attributes of each resource, starting with those explicitly and implicitly mentioned in the prompt, and followed by others: for example, for the aws_efs_replication_configuration, we need to set the "availability_zone_name" attribute to an availability zone that will be within the region specificed in the prompt, such as "us-west-2b". Finally, we connect the resources together, as needed: here "source_file_system_id" should be connected to the "aws_efs_file_system" resource
```hcl
resource "aws_efs_file_system" "example" {}

resource "aws_efs_replication_configuration" "example" {
  source_file_system_id = aws_efs_file_system.example.id

  destination {
    availability_zone_name = "us-west-2b"
    kms_key_id             = "1234abcd-12ab-34cd-56ef-1234567890ab"
  }
}
```

Here is the actual prompt to answer. Let's think step by step:
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": cot_prompt + "Create a template of an elastic beanstalk application" }
]

FastLanguageModel.for_inference(model)
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = 'pt',
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)

model.generate(inputs=inputs, streamer=text_streamer,max_new_tokens=256,use_cache=True)

model.save_pretrained_merged("Qwen2.5-Coder-7B-Instruct-IaC-merged", tokenizer, save_method="merged_16bit")


