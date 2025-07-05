# gpt2_finetune.py
"""
Fine-tune GPT-2 on a custom dataset and generate text based on a prompt.
References:
- https://huggingface.co/blog/how-to-generate
- https://colab.research.google.com/drive/15qBZx5y9rdaQSyWpsreMDnTiZ5IlN0zD?usp=sharing
"""
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import pipeline
import streamlit as st

st.title("GPT-2 Text Generation (Fine-tuned)")
st.write("Fine-tune GPT-2 on your own data and generate text!")

# 1. Download and load GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# File uploader for custom training data
uploaded_file = st.file_uploader("Upload your training text file (train.txt)", type=["txt"])
train_path = os.path.join(os.path.dirname(__file__), 'train.txt')
user_uploaded = False
if uploaded_file is not None:
    with open(train_path, "wb") as f:
        f.write(uploaded_file.read())
    user_uploaded = True
    st.success("train.txt uploaded. You can now fine-tune the model.")
else:
    # If no file uploaded, ensure train.txt exists with default big data
    if not os.path.exists(train_path) or os.path.getsize(train_path) < 1000:
        with open(train_path, "w", encoding="utf-8") as f:
            f.write("""Once upon a time, in a land far, far away, there lived a wise old king who ruled his kingdom with kindness and justice.\nThe sun rises in the east and sets in the west. Every morning, the birds sing their melodious tunes, filling the air with joy.\nArtificial intelligence is transforming the world by enabling machines to learn from data and make intelligent decisions.\nThe quick brown fox jumps over the lazy dog.\nIn the heart of the city, people hustle and bustle, chasing their dreams and ambitions.\nThe rain poured down, drenching the streets and bringing relief to the parched earth.\nShe opened the book and was instantly transported to a magical world filled with adventure.\nTechnology continues to evolve, making our lives easier and more connected than ever before.\nThe chef prepared a delicious meal, blending flavors from around the world.\nHe looked up at the night sky, marveling at the countless stars twinkling above.\nThe scientist conducted experiments to unlock the mysteries of the universe.\nMusic has the power to heal, inspire, and bring people together.\nThe artist painted a masterpiece, capturing the beauty of nature on canvas.\nA gentle breeze rustled the leaves, carrying the scent of blooming flowers.\nThe mountain stood tall and majestic, its peak hidden by clouds.\nChildren laughed and played in the park, their joy infectious.\nThe writer crafted stories that touched the hearts of readers around the world.\nInnovation drives progress, opening new possibilities for the future.\nThe river flowed peacefully, reflecting the colors of the sunset.\nFriendship and love are the greatest treasures one can find.\n""" * 10)
        st.info("No file uploaded. Using default training data.")

# Fine-tune button
if st.button("Fine-tune GPT-2") or (not os.path.exists("./gpt2-finetuned") and not user_uploaded):
    with st.spinner("Fine-tuning in progress. This may take a while..."):
        # 2. Prepare dataset (expects 'train.txt' in the same folder)
        train_path = os.path.join(os.path.dirname(__file__), 'train.txt')
        if not os.path.exists(train_path):
            raise FileNotFoundError("Please provide a 'train.txt' file in Task-1 directory with your training data.")

        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path="train.txt",
            block_size=128
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )

        # 3. Training arguments
        training_args = TrainingArguments(
            output_dir="./gpt2-finetuned",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_steps=10_000,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_steps=100
        )

        # 4. Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset
        )

        # 5. Fine-tune
        trainer.train()

        # 6. Save model
        tokenizer.save_pretrained("./gpt2-finetuned")
        model.save_pretrained("./gpt2-finetuned")
        st.success("Fine-tuning complete! Model saved.")

# Text generation section
st.header("Generate Text")
prompt = st.text_input("Enter a prompt:", "Once upon a time")
max_length = st.slider("Max length", 20, 200, 100)
model_dir = "./gpt2-finetuned"
if not os.path.exists(model_dir):
    st.warning("Fine-tuned model not found. Please fine-tune the model first.")
else:
    if st.button("Generate Text"):
        with st.spinner("Generating text..."):
            try:
                generator = pipeline('text-generation', model=model_dir, tokenizer=model_dir)
                results = generator(prompt, max_length=max_length, num_return_sequences=1)
                st.subheader("Generated Text:")
                st.write(results[0]['generated_text'])
            except Exception as e:
                st.error(f"Error generating text: {e}")
