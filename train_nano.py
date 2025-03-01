from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
import torch
from rich import print
import math
from nanogrpo import GRPO
from streering_vector import SteeringVectorModel

SYSTEM_PROMPT = "Respond in following format:<thinking>{step by step reasoning}</thinking><answer>{number}</answer>"


def prepare_dataset(dataset) -> Dataset:
    extract_hash_answer = (
        lambda text: text.split("####")[1].strip() if "####" in text else None
    )

    def process_example(example: dict):
        answer = extract_hash_answer(example["answer"])
        if answer is None:
            return None
        return {
            "prompt": [
                # {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": answer,
        }

    dataset = dataset.map(
        process_example,
        remove_columns=[
            col for col in dataset.column_names if col not in ["prompt", "answer"]
        ],
    )
    dataset = dataset.filter(lambda x: x is not None)

    return dataset


def reasoning_length_reward(sample: dict, s: str, *args, **kwargs):
    print("-"*100)
    print(s)
    print("-"*10)
    total_reward = 0

    try:
        s = s.split("<｜User｜>")[1]
        s = s.split("</think>")
        s = s[0]
        total_reward += (len(tokenizer.encode(s))/100)
    except:
        return total_reward
    print(total_reward)
    print("-"*100)
    
    return total_reward

dataset = load_dataset("openai/gsm8k", "main")["train"]
dataset = prepare_dataset(dataset)

group_size = 8
micro_group_size =2
lr = 5e-2
weight_decay = 0.1
reward_functions = [
    reasoning_length_reward
]

beta = 0.01
max_new_tokens = 1024
max_iterations = 20
wandb_project = "nanoGRPO-steering-vector"
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# small models are kind of dumb, they need a little push so using this fine-tuned model
# source: https://github.com/joey00072/nanoGRPO/blob/master/cold_start/cold_start_finetune.py
# you can totally use the base model, it will just take longer to converge
# model_name = "joey00072/Llama-3.2-1B-Instruct-cold-start-ft2"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# model_name = "unsloth/Llama-3.2-3B-Instruct"


model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SteeringVectorModel(model)


model = model.to(torch.bfloat16)
print(model)


ref_model = None
trainer = GRPO(
    model,
    ref_model,
    tokenizer=tokenizer,
    group_size=group_size,
    micro_group_size=micro_group_size,
    dataset=dataset,
    reward_functions=reward_functions,
    log_wandb=True,
    wandb_project=wandb_project,
    lr=lr,
    weight_decay=weight_decay,
    beta=beta,
    dtype=torch.bfloat16,
    max_new_tokens=max_new_tokens
)

trainer.train(max_iterations=max_iterations)

torch.save(model.steering_vector.data, "steering_vector.pt")