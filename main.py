import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
try:
    import torch_directml
except ImportError:
    torch_directml = None
import os
import platform
import cpuinfo
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import random

# Define the directory for saving results
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Save the Hugging Face token to avoid authentication issues
from transformers import BertTokenizer, BertModel
from dotenv import load_dotenv
import os

load_dotenv()

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # The following lines are for ensuring deterministic behavior on GPU.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 1. Dataset and Preprocessing
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)


    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_data_loaders(dataset_name="imdb", batch_size=32, train_percentage=10):
    # Define local cache directory
    cache_dir = os.path.join(os.getcwd(), "datasets")
    os.makedirs(cache_dir, exist_ok=True)

    # Load the specified dataset, using the local cache
    print(f"Loading dataset: {dataset_name} from/to cache: {cache_dir}")
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    
    # Determine the correct column names for text and label
    if dataset_name == "imdb":
        text_col, label_col = "text", "label"
    elif dataset_name == "yelp_review_full":
        text_col, label_col = "text", "label"
    elif dataset_name == "amazon_polarity":
        text_col, label_col = "content", "label"
    else:
        # Default to 'text', 'label' but this might fail for other datasets
        text_col, label_col = "text", "label"

    num_classes = dataset['train'].features[label_col].num_classes
    print(f"Dataset has {num_classes} classes.")

    # Use a smaller subset for quicker training
    train_size = int((train_percentage / 100) * len(dataset['train']))
    
    # Ensure test set exists and handle datasets without a predefined 'test' split
    if 'test' in dataset:
        test_split = 'test'
    else:
        # If no test set, create one from the training set
        train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
        dataset['train'] = train_test_split['train']
        dataset['test'] = train_test_split['test']
        test_split = 'test'
        print("No test split found. Created one from the training data.")

    test_limit = min(int(0.2 * train_size), len(dataset[test_split])) # Test on 20% of train size
    test_limit = max(test_limit, 32) # Ensure at least 32 samples for testing
    test_limit = min(test_limit, len(dataset[test_split])) # But don't exceed available samples

    train_subset = dataset['train'].shuffle(seed=42).select(range(train_size))
    test_subset = dataset[test_split].shuffle(seed=42).select(range(test_limit))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = TextDataset(train_subset[text_col], train_subset[label_col], tokenizer)
    test_dataset = TextDataset(test_subset[text_col], test_subset[label_col], tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, num_classes

# 2. RNN Model Definition
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False # Freeze BERT parameters
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        sequence_output = bert_output.last_hidden_state
        rnn_out, _ = self.rnn(sequence_output)
        rnn_out = rnn_out[:, -1, :] # Get last hidden state
        out = self.fc(rnn_out)
        return out

# 3. Training and Evaluation Loop
def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=3, progress_callback=None):
    model.to(device)
    metrics = {
        'epoch_times': [],
        'cpu_util_history': [],
        'mem_util_history': [],
        'train_loss_history': [],
        'val_loss_history': [],
        'val_accuracy_history': []
    }

    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        
        num_batches = len(train_loader)
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Record metrics
            metrics['cpu_util_history'].append(psutil.cpu_percent())
            metrics['mem_util_history'].append(psutil.virtual_memory().percent)

            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if progress_callback:
                progress = (i + 1) / num_batches
                progress_callback(progress, f"Epoch {epoch+1}/{num_epochs} - Batch {i+1}/{num_batches}")

        epoch_end_time = time.time()
        
        # --- Validation Step ---
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_train_loss = loss.item() # A simple approximation of train loss
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = 100 * correct / total

        metrics['train_loss_history'].append(avg_train_loss)
        metrics['val_loss_history'].append(avg_val_loss)
        metrics['val_accuracy_history'].append(val_accuracy)
        # --- End Validation Step ---

        metrics['epoch_times'].append(epoch_end_time - epoch_start_time)

    total_training_time = time.time() - total_start_time
    
    # Aggregate results
    train_metrics = {
        "Total Training Time (s)": total_training_time,
        "Average Epoch Time (s)": sum(metrics['epoch_times']) / num_epochs,
        "Average CPU Utilization (%)": sum(metrics['cpu_util_history']) / len(metrics['cpu_util_history']) if metrics['cpu_util_history'] else 0,
        "Peak CPU Utilization (%)": max(metrics['cpu_util_history']) if metrics['cpu_util_history'] else 0,
        "Average Memory Utilization (%)": sum(metrics['mem_util_history']) / len(metrics['mem_util_history']) if metrics['mem_util_history'] else 0,
        "Peak Memory Utilization (%)": max(metrics['mem_util_history']) if metrics['mem_util_history'] else 0,
    }

    # Add history to the final metrics dictionary
    train_metrics['train_loss_history'] = metrics['train_loss_history']
    train_metrics['val_loss_history'] = metrics['val_loss_history']
    train_metrics['val_accuracy_history'] = metrics['val_accuracy_history']
    train_metrics['cpu_util_history'] = metrics['cpu_util_history']
    train_metrics['mem_util_history'] = metrics['mem_util_history']

    return train_metrics

def test_model(model, test_loader, device, progress_callback=None):
    """Tests the model on the test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    num_batches = len(test_loader)
    
    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if progress_callback:
                progress = (i + 1) / num_batches
                progress_callback(progress, f"Testing - Batch {i+1}/{num_batches}")

    inference_time = time.time() - start_time
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    test_metrics = {
        "Test Loss": avg_loss,
        "Test Accuracy (%)": accuracy
    }
    print(f"Test Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return test_metrics, all_labels, all_preds, inference_time

# 4. Comparison and Visualization
def is_gpu_available():
    """Checks if a compatible GPU is available."""
    try:
        import torch_directml
        return torch_directml.is_available()
    except ImportError:
        return False

def run_cpu_training(learning_rate, num_epochs, batch_size, hidden_layers, hidden_size, train_percentage, dataset_name, save_model=False, train_progress_callback=None, test_progress_callback=None):
    """Runs the training and testing process on the CPU and returns metrics."""
    set_seed() # Set seed for reproducibility
    # Model parameters
    INPUT_SIZE = 768

    train_loader, test_loader, num_classes = get_data_loaders(
        dataset_name=dataset_name, 
        batch_size=batch_size, 
        train_percentage=train_percentage
    )
    
    device = torch.device('cpu')
    model = RNNClassifier(INPUT_SIZE, hidden_size, hidden_layers, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    training_metrics = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=num_epochs, progress_callback=train_progress_callback)
    
    if save_model:
        model_path = os.path.join(RESULTS_DIR, 'cpu_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"CPU model saved to {model_path}")

    test_metrics, true_labels, pred_labels, inference_time = test_model(model, test_loader, device, progress_callback=test_progress_callback)
    
    final_metrics = {**training_metrics, **test_metrics}
    final_metrics['true_labels'] = true_labels
    final_metrics['pred_labels'] = pred_labels
    final_metrics['model_object'] = model # Return the model object itself
    final_metrics['Inference Time (s)'] = inference_time
    return final_metrics

def run_gpu_training(learning_rate, num_epochs, batch_size, hidden_layers, hidden_size, train_percentage, dataset_name, save_model=False, train_progress_callback=None, test_progress_callback=None):
    """Runs the training and testing process on the GPU and returns metrics."""
    set_seed() # Set seed for reproducibility
    # Model parameters
    INPUT_SIZE = 768

    train_loader, test_loader, num_classes = get_data_loaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        train_percentage=train_percentage
    )
    
    device = torch_directml.device()
    model = RNNClassifier(INPUT_SIZE, hidden_size, hidden_layers, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    training_metrics = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=num_epochs, progress_callback=train_progress_callback)

    if save_model:
        model_path = os.path.join(RESULTS_DIR, 'gpu_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"GPU model saved to {model_path}")

    test_metrics, true_labels, pred_labels, inference_time = test_model(model, test_loader, device, progress_callback=test_progress_callback)

    final_metrics = {**training_metrics, **test_metrics}
    final_metrics['true_labels'] = true_labels
    final_metrics['pred_labels'] = pred_labels
    final_metrics['model_object'] = model # Return the model object itself
    final_metrics['Inference Time (s)'] = inference_time
    return final_metrics

def run_quantized_cpu_test(model, batch_size, dataset_name, test_progress_callback=None):
    """Quantizes a trained CPU model and tests its performance."""
    print("Quantizing model...")
    # It's important to ensure the model is on the CPU before quantization
    model.to(torch.device('cpu'))
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.RNN, nn.Linear}, dtype=torch.qint8
    )
    print("Model quantized. Starting test...")

    _, test_loader, _ = get_data_loaders(dataset_name=dataset_name, batch_size=batch_size)
    device = torch.device('cpu')

    # Measure inference time for the quantized model
    test_metrics, true_labels, pred_labels, inference_time = test_model(quantized_model, test_loader, device, progress_callback=test_progress_callback)
    
    final_metrics = {**test_metrics}
    final_metrics['true_labels'] = true_labels
    final_metrics['pred_labels'] = pred_labels
    final_metrics['Inference Time (s)'] = inference_time
    
    return final_metrics


def load_and_test_model(model_path, device_type, batch_size, hidden_layers, hidden_size, dataset_name, test_progress_callback=None):
    """Loads a saved model and tests its performance."""
    INPUT_SIZE = 768
    
    full_model_path = os.path.join(RESULTS_DIR, model_path)
    if not os.path.exists(full_model_path):
        return {"Error": f"Model file not found at {full_model_path}."}

    _, test_loader, num_classes = get_data_loaders(dataset_name=dataset_name, batch_size=batch_size)
    
    if device_type == 'gpu' and is_gpu_available():
        device = torch_directml.device()
    else:
        device = torch.device('cpu')

    model = RNNClassifier(INPUT_SIZE, hidden_size, hidden_layers, num_classes)
    model.load_state_dict(torch.load(full_model_path, map_location=device))
    model.to(device)

    print(f"Testing model {full_model_path} on {device}...")
    test_metrics, true_labels, pred_labels, inference_time = test_model(model, test_loader, device, progress_callback=test_progress_callback)
    
    final_metrics = {**test_metrics}
    final_metrics['true_labels'] = true_labels
    final_metrics['pred_labels'] = pred_labels
    final_metrics['Inference Time (s)'] = inference_time
    return final_metrics

def generate_confusion_matrix_plot(true_labels, pred_labels):
    """Generates a confusion matrix plot."""
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    return fig

def generate_comparison(cpu_metrics, gpu_metrics, quantized_cpu_metrics=None, report_dir=None):
    """Generates the comparison dataframe and chart."""
    metrics_to_plot = [
        "Total Training Time (s)", "Inference Time (s)", "Average Epoch Time (s)",
        "Test Loss", "Test Accuracy (%)",
        "Average CPU Utilization (%)", "Peak CPU Utilization (%)",
        "Average Memory Utilization (%)", "Peak Memory Utilization (%)"
    ]
    
    df_cpu = pd.DataFrame.from_dict(cpu_metrics, orient='index', columns=['CPU'])
    df_gpu = pd.DataFrame.from_dict(gpu_metrics, orient='index', columns=['GPU'])
    
    df_combined = pd.concat([df_cpu, df_gpu], axis=1)

    if quantized_cpu_metrics:
        # For quantized model, we care about inference time, not training time
        quantized_metrics_for_df = {
            "Inference Time (s)": quantized_cpu_metrics.get("Inference Time (s)"),
            "Test Loss": quantized_cpu_metrics.get("Test Loss"),
            "Test Accuracy (%)": quantized_cpu_metrics.get("Test Accuracy (%)")
        }
        df_quantized = pd.DataFrame.from_dict(quantized_metrics_for_df, orient='index', columns=['Quantized CPU'])
        df_combined = pd.concat([df_combined, df_quantized], axis=1)


    df_to_plot = df_combined.loc[df_combined.index.intersection(metrics_to_plot)]

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    df_to_plot.plot(kind='bar', ax=ax, rot=45)
    ax.set_title('CPU vs GPU Training Performance Comparison')
    ax.set_ylabel('Values')
    plt.tight_layout()

    # Determine save path
    if report_dir:
        chart_path = os.path.join(report_dir, 'performance_comparison.png')
    else:
        chart_path = os.path.join(RESULTS_DIR, 'performance_comparison.png')
    
    fig.savefig(chart_path)
    plt.close(fig)
    
    return df_to_plot, chart_path

def get_system_info():
    """Gathers and returns key system information."""
    info = {}
    try:
        info['OS'] = f"{platform.system()} {platform.release()}"
        info['Python Version'] = platform.python_version()
        
        # CPU Info
        info['CPU'] = cpuinfo.get_cpu_info()['brand_raw']
        
        # RAM Info
        ram_gb = psutil.virtual_memory().total / (1024**3)
        info['Total RAM'] = f"{ram_gb:.2f} GB"
        
        # GPU Info
        if is_gpu_available():
            # torch-directml does not provide a simple way to get the device name.
            # We will confirm that a DirectML-compatible device is being used.
            info['GPU'] = "DirectML-compatible GPU Detected"
        else:
            info['GPU'] = "Not Available / Not Detected"
            
    except Exception as e:
        info['Error'] = f"Could not retrieve all system info: {e}"
        
    return info

def get_dataset_preview(dataset_name):
    """Loads a preview of the selected dataset."""
    try:
        # Define local cache directory
        cache_dir = os.path.join(os.getcwd(), "datasets")
        dataset = load_dataset(dataset_name, split='train[:5%]', cache_dir=cache_dir)
        df = pd.DataFrame(dataset)
        return df.head()
    except Exception as e:
        print(f"Error loading dataset preview: {e}")
        return None

# Main execution (for running as a standalone script)
if __name__ == "__main__":
    cpu_metrics = run_cpu_training()
    
    if is_gpu_available():
        gpu_metrics = run_gpu_training()
        generate_comparison(cpu_metrics, gpu_metrics)
        # Print to console
        df, _ = generate_comparison(cpu_metrics, gpu_metrics)
        print("\n--- Comparison Metrics ---")
        print(df)
    else:
        print("\nGPU not available. Skipping GPU training and comparison.")
        print("\nCPU Metrics:")
        for key, value in cpu_metrics.items():
            print(f"{key}: {value}")
