from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chatbot_response(user_input):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

print(chatbot_response("Hello, how are you?"))
