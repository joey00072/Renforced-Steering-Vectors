from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from streering_vector import SteeringVectorModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

steering_vector = torch.load("steering_vector.pt")  

model = SteeringVectorModel(model, steering_vector)
model = model.to(device)
streamer = TextStreamer(tokenizer)
max_new_tokens = 1024 * 2
prompt = "If $f(x) = \frac{3x-2}{x-2}$, what is the value of $f(-2) +f(-1)+f(0)$? Express your answer as a common fraction."
input_ids = tokenizer.encode(prompt, return_tensors="pt")
input_ids = input_ids.to(device)
model.generate(input_ids, streamer=streamer, max_new_tokens=max_new_tokens , temperature=0.9)




