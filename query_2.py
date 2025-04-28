from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datetime import datetime
import csv
import re

def query_model(prompt, model_dir="./models/gpt-j-6B", max_length=500):
    print("Initializing model loading process...")
    try:
        # Check if model exists in local directory
        if os.path.exists(model_dir):
            print(f"Loading saved model from: {model_dir}")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(model_dir)
        else:
            print("Downloading GPT-J model for the first time (this may take a while)...")
            tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
            model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
            
            # Save the model locally
            print(f"Saving model to: {model_dir}")
            os.makedirs(model_dir, exist_ok=True)
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)
            print("Model saved successfully!")

    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
    
    # Encode the prompt into input tokens
    print("Encoding input...")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text using the model
    print("Generating response...")
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text[:200]}...")  # Print first 200 characters
    return generated_text

def save_response_to_file(response, folder="responses"):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Generate a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder}/response_{timestamp}.txt"
    
    # Save the response to the file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response)
    
    return filename

def clean_data_to_csv(text_data, output_file='cleaned_profiles.csv'):
    # Split the text into individual profiles
    profiles = re.split(r'-{5,}', text_data)
    profiles = [p.strip() for p in profiles if p.strip() and not p.strip().startswith('Please clean this data')]
    
    parsed_data = []
    for profile in profiles:
        data = {}
        
        # Extract name
        name_match = re.search(r'Name:\s*(.*?)(?=EMAIL:|$)', profile, re.DOTALL)
        if name_match:
            data['name'] = re.sub(r'\s+', ' ', name_match.group(1).strip())
        
        # Extract email
        email_match = re.search(r'EMAIL:\s*(.*?)(?=Date Created:|$)', profile, re.DOTALL)
        if email_match:
            data['email'] = email_match.group(1).strip()
        
        # Extract date
        date_match = re.search(r'Date Created:\s*(.*?)(?=BIO TEXT:|$)', profile, re.DOTALL)
        if date_match:
            data['date_created'] = re.sub(r'\s+', ' ', date_match.group(1).strip())
        
        # Extract bio
        bio_match = re.search(r'BIO TEXT:(.*?)(?=Current address:|$)', profile, re.DOTALL)
        if bio_match:
            data['bio'] = re.sub(r'\s+', ' ', bio_match.group(1).strip())
        
        # Extract address
        address_match = re.search(r'Current address:(.*?)(?=---|$)', profile, re.DOTALL)
        if address_match:
            data['address'] = re.sub(r'\s+', ' ', address_match.group(1).strip())
        
        parsed_data.append(data)
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'email', 'date_created', 'bio', 'address']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(parsed_data)
    
    return f"Successfully converted data to {output_file}"


def main():
    prompt = input("Enter your prompt: ")
    response = query_model(prompt)
    print("\nResponse:")
    print(response)
    
    # Save response to file
    saved_file = save_response_to_file(response)
    print(f"\nResponse saved to: {saved_file}")
    
    # Ask if user wants to convert response to CSV
    convert_to_csv = input("\nDo you want to convert this response to CSV? (y/n): ")
    if convert_to_csv.lower() == 'y':
        # Generate CSV filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"cleaned_profiles_{timestamp}.csv"
        
        # Convert to CSV
        result = clean_data_to_csv(response, csv_filename)
        print(result)

if __name__ == "__main__":
    main()