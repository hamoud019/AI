"""
Download aragpt2-base model for offline use on Kaggle.
Run this locally: python download_model.py
Then upload the 'aragpt2-base-local' folder as a Kaggle dataset.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Downloading aubmindlab/aragpt2-base...")
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/aragpt2-base")
model = AutoModelForCausalLM.from_pretrained("aubmindlab/aragpt2-base")

save_dir = "aragpt2-base-local"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
print(f"Model saved to '{save_dir}/' - Upload this folder as a Kaggle dataset!")
