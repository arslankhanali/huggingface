from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
model = AutoModelForCausalLM.from_pretrained(
  "stabilityai/stablelm-3b-4e1t",
  trust_remote_code=True,
  torch_dtype="auto",
)
model.cpu()
inputs = tokenizer("Lion was lonely", return_tensors="pt").to("cpu")
tokens = model.generate(
  **inputs,
  max_new_tokens=64,
  temperature=0.75,
  top_p=0.95,
  do_sample=True,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))