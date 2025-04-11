from transformers import GPT2Tokenizer, GPT2LMHeadModel

def query_model(prompt, model_dir="./gpt2_finetuned", max_length=100):
    # Load the tokenizer and model from your fine-tuned directory
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    print(f"Model loaded from: {model_dir}")
    
    # Encode the prompt into input tokens
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Input IDs: {input_ids}")
    
    # Generate text using the model
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,          # Use sampling for more creative generations
        temperature=0.7,         # Lower values make the model more deterministic
        top_p=0.9,               # Use nucleus sampling
        num_return_sequences=1,  # Generate a single response
        pad_token_id=tokenizer.eos_token_id,
    )
    print(f"Output IDs: {output_ids}")
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text[:200]}...")  # Print first 200 characters for brevity
    return generated_text

def main():
    prompt = input("Enter your prompt: ")
    response = query_model(prompt)
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()