from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
)
from data_loader import load_texts_from_csv, save_texts_to_file

def main():
    # Path to your CSV file containing raw text data
    csv_file = "raw_data.csv"

    # Load texts from CSV and save them to a text file
    texts = load_texts_from_csv(csv_file)
    dataset_file = save_texts_to_file(texts, txt_path="dataset.txt")
    print(f"Dataset file created at: {dataset_file}")

    # Load the saved model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("./saved_gpt2_model")
    model = GPT2LMHeadModel.from_pretrained("./saved_gpt2_model")
    print("Loaded saved model and tokenizer")

    # Create a dataset using the generated text file
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=dataset_file,
        block_size=128,  # adjust block size as needed
    )

    # Create a data collator for language modeling (without masked LM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    print("Data collator created with block size: 128")

    training_args = TrainingArguments(
        output_dir="./gpt2_finetuned",
        overwrite_output_dir=True,
        num_train_epochs=1,  # reduce epochs
        per_device_train_batch_size=8,  # try increasing batch size if possible
        gradient_accumulation_steps=2,  # accumulate gradients over steps
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )
    print(f"Training arguments: {training_args}")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    print("Starting training...")

    trainer.train()
    trainer.save_model("./gpt2_finetuned")
    print("Model training complete and saved to ./gpt2_finetuned")

if __name__ == "__main__":
    main()