import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and Qwen3-0.6B model
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

qwen_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
qwen_model.eval()

# Load datasets
original_dataset = load_dataset("glue", "sst2")
val_dataset_original = original_dataset["validation"]

all_predictions = []
all_labels = []

print("Evaluating Qwen3-0.6B on SST-2 Classification Task (Prompt-Based Generation)...")
print("="*60)

with torch.no_grad():
    for idx in tqdm(range(len(val_dataset_original)), desc="Generating Predictions"):
        sentence = val_dataset_original[idx]["sentence"]
        true_label = val_dataset_original[idx]["label"]
        
        # Create prompt asking for sentiment classification
        prompt = f"""Analyze the sentiment of the following sentence and respond with only one word: either "positive" or "negative".

Sentence: {sentence}

Sentiment:"""
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        # Generate response
        output_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=10,  # Only need a few tokens for "positive" or "negative"
            do_sample=False,    # Deterministic generation
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode only the generated part (exclude the prompt)
        generated_text = tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip().lower()
        
        # Parse the response to extract sentiment
        # Look for "positive" or "negative" in the generated text
        if "positive" in generated_text:
            prediction = 1
        elif "negative" in generated_text:
            prediction = 0
        else:
            # Fallback: if response is ambiguous, check first word
            first_word = generated_text.split()[0] if generated_text.split() else ""
            if first_word.startswith("pos"):
                prediction = 1
            elif first_word.startswith("neg"):
                prediction = 0
            else:
                # Random guess if completely unclear
                prediction = 0
        
        all_predictions.append(prediction)
        all_labels.append(true_label)

# Convert to numpy arrays
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_predictions, average='binary', zero_division=0
)

cm = confusion_matrix(all_labels, all_predictions)
class_accuracies = cm.diagonal() / cm.sum(axis=1)

# Print results
print("\n" + "="*60)
print("QWEN3-0.6B CLASSIFICATION RESULTS ON SST-2 (PROMPT-BASED)")
print("="*60)

print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print(f"\n### Per-Class Performance ###")
print(f"Negative Class (0) Accuracy: {class_accuracies[0]:.4f} ({class_accuracies[0]*100:.2f}%)")
print(f"Positive Class (1) Accuracy: {class_accuracies[1]:.4f} ({class_accuracies[1]*100:.2f}%)")

print(f"\n### Confusion Matrix ###")
print(f"                Predicted")
print(f"              Neg    Pos")
print(f"Actual Neg  [{cm[0,0]:5d}  {cm[0,1]:5d}]")
print(f"       Pos  [{cm[1,0]:5d}  {cm[1,1]:5d}]")

print(f"\n### Dataset Statistics ###")
print(f"Total Samples: {len(all_labels)}")
print(f"Positive Samples: {np.sum(all_labels == 1)} ({np.sum(all_labels == 1)/len(all_labels)*100:.2f}%)")
print(f"Negative Samples: {np.sum(all_labels == 0)} ({np.sum(all_labels == 0)/len(all_labels)*100:.2f}%)")

print("="*60)
