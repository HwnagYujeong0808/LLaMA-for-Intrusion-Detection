import numpy as np
import os
import time
import random
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import math
import sys
from datetime import datetime
import pytz
import wandb
import sys
import os
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

class Logger:
    def __init__(self, filename="output.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')  # Use 'a' to append to the log file

    def write(self, message):
        # Write to console
        self.terminal.write(message)
        self.terminal.flush()
        # Write to log file
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("output.log")
sys.stderr = sys.stdout  # Redirect stderr to the same logger


def print_log(message):
    """Prints a message with a single timestamp in the ETC timezone."""
    timezone = pytz.timezone("Etc/GMT")
    timestamp = datetime.now(timezone).strftime("[%Y-%m-%d %H:%M:%S] ")
    print(f"{timestamp} {message}")


# Redirect stdout to Logger
sys.stdout = Logger("output.log")

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LLMGraphTransformer(nn.Module):
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", device="cpu"):
        super().__init__()
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.dropout = nn.Dropout(p=0.2)
        self.edge_fc = nn.Linear(77, 64).to(self.device)
        self.edge_dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(64 + 64, 9).to(self.device)
    
    def forward(self, batch_text, edge_features):
        inputs = self.tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        outputs = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        text_logits = outputs.logits[:, -1, :]
        text_emb = self.dropout(text_logits)
        text_emb = text_emb[:, :64]
        edge_emb = self.edge_fc(edge_features)
        edge_emb = self.edge_dropout(edge_emb)
        combined_emb = torch.cat((text_emb, edge_emb), dim=1)
        output = self.classifier(combined_emb)
        return output

    def generate_text(self, batch_edges, labels):
        batch_text = []
        for edge in batch_edges:
            node1, node2 = edge
            question = f"What is the relationship between Node {node1} and Node {node2}? Choices: {', '.join(labels)}."
            batch_text.append(question)
        return batch_text

wandb.login(key="445490c10d80ab2617dc067c4fa08485255f55d9")

wandb.init(
    project="Intrusion-Detection",  # Set your project name
    config={
        "learning_rate": 1e-5,
        "epochs": 10,
        "batch_size": 8,
        "accumulation_steps": 4,
    },
    settings=wandb.Settings(console="off")
)



def fit(args):
    data = args["dataset"]
    path = "datasets/" + data
    if not path.endswith('/'):
        path += '/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edge_feat = np.load(os.path.join(path, "edge_feat_scaled.npy"), allow_pickle=True)
    edge_feat = torch.tensor(edge_feat, dtype=torch.float, device=device)

    label = np.load(os.path.join(path, "label_mul.npy"), allow_pickle=True)
    label = torch.tensor(label, dtype=torch.long, device=device)

    edge_list = np.load(os.path.join(path, "adj_random.npy"), allow_pickle=True)

    llm_graph_transformer = LLMGraphTransformer(model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", device=device)
    labels = ['Normal', 'Audio-Streaming', 'Browsing', 'Chat', 'File-Transfer', 'Email', 'P2P', 'Video-Streaming', 'VOIP']
    
    unique_labels = np.unique(label.cpu().numpy())
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=label.cpu().numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Use weighted loss function
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(llm_graph_transformer.parameters(), lr=1e-5)
     # Compute class weights

    num_samples = len(edge_feat)
    data_indices = np.arange(num_samples)
    train_val, test = train_test_split(data_indices, test_size=0.1, random_state=42, stratify=label.cpu())
    train, val = train_test_split(train_val, test_size=0.1, random_state=42, stratify=label[train_val].cpu())

    batch_size = 8
    num_batches = int(np.ceil(len(train) / batch_size))
    accumulation_steps = 4

    # Early Stopping Settings
    best_val_loss = float('inf')
    patience = 3
    trigger_times = 0
    best_model_state = None
    best_optimizer_state = None

    for epoch in range(10):
        print_log(f"\nEpoch {epoch + 1}")
        llm_graph_transformer.train()
        random.shuffle(train)
        total_loss = 0
        all_true_labels = []
        all_predicted_labels = []

        # Training Loop
        for batch_idx in range(num_batches):
            batch_indices = train[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_edges = edge_list[batch_indices]
            batch_text = llm_graph_transformer.generate_text(batch_edges, labels)

            edge_batch = edge_feat[batch_indices]
            batch_labels = label[batch_indices].to(device)

            logits = llm_graph_transformer(batch_text, edge_batch)
            loss = loss_fn(logits, batch_labels) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                torch.nn.utils.clip_grad_norm_(llm_graph_transformer.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Batch-wise evaluation
            predicted_labels = torch.argmax(logits, dim=-1).cpu().numpy()
            all_true_labels.extend(batch_labels.cpu().numpy())
            all_predicted_labels.extend(predicted_labels)

            batch_f1 = f1_score(batch_labels.cpu().numpy(), predicted_labels, average="weighted")
            total_loss += loss.item() * accumulation_steps

            # Log batch-level metrics to W&B for real-time visualization
            wandb.log({
                "batch_loss": loss.item() * accumulation_steps,
                "batch_f1_score": batch_f1,
                "epoch": epoch + 1,
                "batch": batch_idx + 1,
            })

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print_log(f"Batch {batch_idx + 1}/{num_batches} - Loss: {loss.item():.4f}, F1 Score: {batch_f1:.4f}")

        avg_loss = total_loss / num_batches
        avg_f1 = f1_score(all_true_labels, all_predicted_labels, average="weighted")
        print_log(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, Avg F1 Score: {avg_f1:.4f}")

        # Log epoch-level metrics to W&B
        wandb.log({
            "train_loss": avg_loss,
            "train_f1_score": avg_f1,
            "epoch": epoch + 1,
        })

        # Validation Loop
        val_loss, val_f1, val_label_acc = evaluate(llm_graph_transformer, edge_list, edge_feat, label, val, labels, loss_fn, device)
        print_log(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}, Validation F1 Score: {val_f1:.4f}, Label Accuracy: {val_label_acc}")

        # Log validation metrics to W&B
        wandb.log({
            "val_loss": val_loss,
            "val_f1_score": val_f1,
            "val_label_accuracy": val_label_acc,
            "epoch": epoch + 1,
        })

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # Save the best model state
            best_model_state = llm_graph_transformer.state_dict()
            best_optimizer_state = optimizer.state_dict()
        else:
            trigger_times += 1
            print_log(f"No improvement in validation loss for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                print_log("Early stopping triggered.")
                break

    # Save the best model at the end
    if best_model_state is not None and best_optimizer_state is not None:
        llm_graph_transformer.load_state_dict(best_model_state)
        optimizer.load_state_dict(best_optimizer_state)
        save_model(llm_graph_transformer, optimizer, "final")

    # Test Loop
    test_loss, test_f1 = evaluate(llm_graph_transformer, edge_list, edge_feat, label, test, labels, loss_fn, device)
    print_log(f"Test Loss: {test_loss:.4f}, Test F1 Score: {test_f1:.4f}")

    # Log test metrics to W&B
    wandb.log({
        "test_loss": test_loss,
        "test_f1_score": test_f1,
    })

    # Finish the W&B run
    wandb.finish()



def evaluate(model, edge_list, edge_feat, label, data_indices, labels, loss_fn, device):
    model.eval()
    total_loss = 0
    all_true_labels = []
    all_predicted_labels = []
    with torch.no_grad():
        for i in range(0, len(data_indices), 8):  # Batch size of 8
            batch_indices = data_indices[i:i + 8]
            batch_edges = edge_list[batch_indices]
            batch_text = model.generate_text(batch_edges, labels)

            edge_batch = edge_feat[batch_indices]
            batch_labels = label[batch_indices].to(device)

            logits = model(batch_text, edge_batch)
            loss = loss_fn(logits, batch_labels)
            predicted_labels = torch.argmax(logits, dim=-1).cpu().numpy()

            all_true_labels.extend(batch_labels.cpu().numpy())
            all_predicted_labels.extend(predicted_labels)
            total_loss += loss.item() * len(batch_indices)

    avg_loss = total_loss / len(data_indices)
    avg_f1 = f1_score(all_true_labels, all_predicted_labels, average="weighted")
    label_acc = f1_score(all_true_labels, all_predicted_labels, average=None)
    return avg_loss, avg_f1, label_acc


def save_model(model, optimizer, epoch, path="llm.pth"):
    current_time = time.strftime("%Y%m%d-%H%M%S")
    path = f"model/{current_time}_{path}"
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # Create directory if it does not exist
    os.makedirs("model", exist_ok=True)
    torch.save(checkpoint, path)
    
    print_log(f"Model saved to {path}")

if __name__ == "__main__":
    set_seeds(42)
    fit({
        "dataset": "Darknet",
        "binary": False,
    })

