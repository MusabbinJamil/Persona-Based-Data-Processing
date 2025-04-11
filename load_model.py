from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_and_save_model():
    print("Loading model and tokenizer...")
    # Load tokenizer and add pad token if it is missing
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Load GPT-2 model and update token embeddings
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    print("Model and tokenizer loaded successfully.")
    
    # Save the model and tokenizer
    model.save_pretrained("./saved_gpt2_model")
    tokenizer.save_pretrained("./saved_gpt2_model")
    print("Model saved to ./saved_gpt2_model")

if __name__ == "__main__":
    load_and_save_model()