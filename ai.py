from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pre-trained model
model_name = "microsoft/DialoGPT-small"  # A conversational model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(user_input):
    # Tokenize input and generate response
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)

# Chat loop
print("Chatbot: Hello! Type 'bye' to exit.")
while True:
    user_message = input("You: ")
    if user_message.lower() == "bye":
        print("Chatbot: Goodbye!")
        break
    response = generate_response(user_message)
    print(f"Chatbot: {response}")
