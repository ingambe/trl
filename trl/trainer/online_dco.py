from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import re
import bitsandbytes as bnb
import types
import torch
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM


def format_reward_func(completions, ground_truth, **kwargs):
    rewards = []
    
    for completion, gt in zip(completions, ground_truth):
        content = completion[0]["content"]

        # Find all boxed occurrences
        matches = re.findall(r"\\boxed\{(.*?)\}", content)

        # Reward 1 if there is exactly one \boxed{} and its content matches ground truth
        reward = 1.0 if len(matches) == 1 and matches[0] == gt else 0.0
        rewards.append(reward)

    return rewards


def reward_func(completions, ground_truth, **kwargs):
    #print(completions)
    #print(ground_truth)
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion[0]["content"]) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]
    
dataset = load_dataset("ai2-adapt-dev/gsm8k_math_ground_truth", split="train")

# Preprocessing: rename messages to prompt and add a system prompt
def preprocess(example):
    system_prompt = {
        "role": "system", 
        "content": "Reason step by step, and put your final answer within \\boxed{{}}."
    }
    example["prompt"] = [system_prompt] + example["messages"]
    
    example["completion"] = [{
        "role": "assistant", 
        "content": ""
    }]
    example["ground_truth"] = example.get("ground_truth", "")
    
    return example

dataset = dataset.map(preprocess).remove_columns(["messages"])

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", 
    logging_steps=8,
    max_grad_norm=0.1,
    beta=0,
    save_steps=128,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_prompt_length=700,
    max_completion_length=400,
    num_generations=8,
    vllm_max_model_len=1024,
    #use_vllm=True,
    report_to="wandb",
    log_completions=True)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

optimizer = bnb.optim.PagedAdamW8bit(
    model.parameters(),
    lr=1e-6,
    betas=(0.9, 0.95),
    weight_decay=0.1
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=training_args.num_train_epochs * len(dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * torch.cuda.device_count())
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward_func, reward_func],
    args=training_args,
    train_dataset=dataset,
    optimizers=(optimizer, scheduler)
)

trainer.train()