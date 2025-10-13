import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and Qwen3-0.6B model
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Qwen3 uses <|endoftext|> as pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

qwen_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
qwen_model.eval()

# Load the original SST-2 dataset to get labels
original_dataset = load_dataset("glue", "sst2")
val_dataset_original = original_dataset["validation"]

# Load tokenized dataset
val_dataset_tokenized = load_from_disk("/kaggle/working/tokenized_sst2_qwen3/validation")

# Combine: use tokenized data for input_ids, original data for labels
# Make sure they're aligned (same order, same length)
assert len(val_dataset_tokenized) == len(val_dataset_original), "Dataset length mismatch!"

val_loader = DataLoader(val_dataset_tokenized, batch_size=16, shuffle=False)

# Define sentiment keywords/tokens for zero-shot classification
positive_token_id = tokenizer.encode(" positive", add_special_tokens=False)[0]
negative_token_id = tokenizer.encode(" negative", add_special_tokens=False)[0]

all_predictions = []
all_labels = []

print("Evaluating Qwen3-0.6B on SST-2 Classification Task...")
print("="*60)

sample_idx = 0

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Computing Predictions"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        batch_size = input_ids.size(0)
        
        # Get corresponding labels from original dataset
        labels = [val_dataset_original[sample_idx + i]["label"] for i in range(batch_size)]
        sample_idx += batch_size
        
        # Get model outputs
        outputs = qwen_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Get predictions for the last non-padding token
        batch_predictions = []
        
        for i in range(input_ids.size(0)):
            # Find the last non-pad token position
            non_pad_positions = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
            if len(non_pad_positions) > 0:
                last_pos = non_pad_positions[-1].item()
            else:
                last_pos = 0
            
            # Get logits for the last position
            last_token_logits = logits[i, last_pos, :]
            
            # Compare probabilities for positive vs negative tokens
            pos_prob = torch.softmax(last_token_logits, dim=-1)[positive_token_id].item()
            neg_prob = torch.softmax(last_token_logits, dim=-1)[negative_token_id].item()
            
            # Predict based on which has higher probability
            if pos_prob > neg_prob:
                prediction = 1  # Positive sentiment
            else:
                prediction = 0  # Negative sentiment
            
            batch_predictions.append(prediction)
        
        all_predictions.extend(batch_predictions)
        all_labels.extend(labels)

# Convert to numpy arrays
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_predictions, average='binary', zero_division=0
)

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Per-class accuracy
class_accuracies = cm.diagonal() / cm.sum(axis=1)

# Print results
print("\n" + "="*60)
print("QWEN3-0.6B CLASSIFICATION RESULTS ON SST-2")
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
