import streamlit as st
import pandas as pd
import main as training_script
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="CPU vs. GPU Training", layout="wide")

st.title("🔬 RNN Training Performance: CPU vs. GPU")

with st.expander("💻 System Information", expanded=True):
    sys_info = training_script.get_system_info()
    col1, col2 = st.columns(2)
    items = list(sys_info.items())
    mid_index = (len(items) + 1) // 2
    
    with col1:
        for key, value in items[:mid_index]:
            st.markdown(f"**{key}:**")
            st.write(value)
    with col2:
        for key, value in items[mid_index:]:
            st.markdown(f"**{key}:**")
            st.write(value)

st.sidebar.header("Hyperparameters")
dataset_name = st.sidebar.selectbox("Select Dataset", ["imdb", "yelp_review_full", "amazon_polarity"])

with st.sidebar.expander("Dataset Preview"):
    preview_df = training_script.get_dataset_preview(dataset_name)
    st.dataframe(preview_df, use_container_width=True)

learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
num_epochs = st.sidebar.slider("Number of Epochs", 1, 10, 1)
batch_size = st.sidebar.select_slider("Batch Size", options=[8, 16, 32, 64], value=16)
hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 2)
hidden_size = st.sidebar.select_slider("Hidden Layer Size", options=[128, 256, 512], value=256)
train_percentage = st.sidebar.slider("Training Dataset Size (%)", 1, 100, 10)
save_models_checkbox = st.sidebar.checkbox("Save Trained Models", value=False)

st.write("""
This application demonstrates the performance difference between training a Recurrent Neural Network (RNN) on a CPU versus a GPU. 
Click the button below to start the training process on both devices and see the results.
""")

if 'cpu_metrics' not in st.session_state:
    st.session_state.cpu_metrics = None
if 'gpu_metrics' not in st.session_state:
    st.session_state.gpu_metrics = None
if 'comparison_df' not in st.session_state:
    st.session_state.comparison_df = None
if 'chart_path' not in st.session_state:
    st.session_state.chart_path = None
if 'quantized_metrics' not in st.session_state:
    st.session_state.quantized_metrics = None

def plot_history(metrics, device_name):
    """Plots the training and validation history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(metrics['train_loss_history']) + 1)
    
    # Plot Loss
    ax1.plot(epochs, metrics['train_loss_history'], 'b-o', label='Training Loss')
    ax1.plot(epochs, metrics['val_loss_history'], 'r-o', label='Validation Loss')
    ax1.set_title(f'{device_name} - Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot Accuracy
    ax2.plot(epochs, metrics['val_accuracy_history'], 'g-o', label='Validation Accuracy')
    ax2.set_title(f'{device_name} - Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_utilization_history(metrics, device_name):
    """Plots the CPU and Memory utilization over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # CPU Utilization
    if 'cpu_util_history' in metrics and metrics['cpu_util_history']:
        ax.plot(metrics['cpu_util_history'], label='CPU Utilization (%)', color='orange')
    
    # Memory Utilization
    if 'mem_util_history' in metrics and metrics['mem_util_history']:
        ax.plot(metrics['mem_util_history'], label='Memory Utilization (%)', color='purple')
        
    ax.set_title(f'{device_name} - Resource Utilization During Training')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Utilization (%)')
    ax.legend()
    plt.tight_layout()
    return fig

if st.button("🚀 Start Training and Comparison"):
    # Create a unique directory for this run's results
    scenario_name = f"scenario_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    report_dir = os.path.join(training_script.RESULTS_DIR, scenario_name)
    os.makedirs(report_dir, exist_ok=True)
    st.session_state.report_dir = report_dir

    # Save hyperparameters to a file
    hyperparameters = {
        "Dataset": dataset_name,
        "Learning Rate": learning_rate,
        "Epochs": num_epochs,
        "Batch Size": batch_size,
        "Hidden Layers": hidden_layers,
        "Hidden Size": hidden_size,
        "Training Percentage": train_percentage
    }
    with open(os.path.join(report_dir, 'hyperparameters.txt'), 'w') as f:
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")

    st.session_state.cpu_metrics = None
    st.session_state.gpu_metrics = None
    st.session_state.comparison_df = None
    st.session_state.chart_path = None
    st.session_state.quantized_metrics = None

    # --- CPU Training ---
    st.write("### 🏃‍♂️ Training on CPU...")
    cpu_chart_placeholder = st.empty()
    train_progress_bar_cpu = st.progress(0)
    train_status_text_cpu = st.empty()
    def cpu_train_progress_callback(progress, text):
        train_progress_bar_cpu.progress(progress)
        train_status_text_cpu.text(text)
    
    st.write("### 🧪 Testing on CPU...")
    test_progress_bar_cpu = st.progress(0)
    test_status_text_cpu = st.empty()
    def cpu_test_progress_callback(progress, text):
        test_progress_bar_cpu.progress(progress)
        test_status_text_cpu.text(text)

    st.session_state.cpu_metrics = training_script.run_cpu_training(
        learning_rate, num_epochs, batch_size, hidden_layers, hidden_size,
        train_percentage=train_percentage,
        dataset_name=dataset_name,
        save_model=save_models_checkbox,
        train_progress_callback=cpu_train_progress_callback,
        test_progress_callback=cpu_test_progress_callback
    )
    train_status_text_cpu.text("✅ CPU training complete!")
    test_status_text_cpu.text("✅ CPU testing complete!")
    st.success("✅ CPU process complete!")
    st.write("### 📊 CPU Performance Metrics")
    st.json({k: v for k, v in st.session_state.cpu_metrics.items() if not isinstance(v, list)})
    
    # Plot CPU history
    if 'train_loss_history' in st.session_state.cpu_metrics:
        st.write("#### CPU Training History")
        cpu_fig = plot_history(st.session_state.cpu_metrics, "CPU")
        cpu_chart_placeholder.pyplot(cpu_fig)
        cpu_fig.savefig(os.path.join(st.session_state.report_dir, 'cpu_training_history.png'))

    # Plot CPU utilization
    if 'cpu_util_history' in st.session_state.cpu_metrics:
        st.write("#### CPU Resource Utilization")
        cpu_util_fig = plot_utilization_history(st.session_state.cpu_metrics, "CPU")
        st.pyplot(cpu_util_fig)
        cpu_util_fig.savefig(os.path.join(st.session_state.report_dir, 'cpu_resource_utilization.png'))

    # Plot Confusion Matrix for CPU
    if 'true_labels' in st.session_state.cpu_metrics:
        st.write("#### CPU Confusion Matrix")
        cm_fig_cpu = training_script.generate_confusion_matrix_plot(
            st.session_state.cpu_metrics['true_labels'],
            st.session_state.cpu_metrics['pred_labels']
        )
        st.pyplot(cm_fig_cpu)
        cm_fig_cpu.savefig(os.path.join(st.session_state.report_dir, 'cpu_confusion_matrix.png'))

    # --- Quantized CPU Testing ---
    st.write("### 🧠 Quantizing and Testing on CPU...")
    quant_progress_bar = st.progress(0)
    quant_status_text = st.empty()
    def quant_test_progress_callback(progress, text):
        quant_progress_bar.progress(progress)
        quant_status_text.text(text)

    # Use the model object returned from the CPU training run
    if 'model_object' in st.session_state.cpu_metrics:
        cpu_model_for_quant = st.session_state.cpu_metrics['model_object']
        st.session_state.quantized_metrics = training_script.run_quantized_cpu_test(
            model=cpu_model_for_quant,
            batch_size=batch_size,
            dataset_name=dataset_name,
            test_progress_callback=quant_test_progress_callback
        )
        st.success("✅ Quantized CPU testing complete!")
        st.write("### 🧠 Quantized CPU Performance")
        st.json({k: v for k, v in st.session_state.quantized_metrics.items() if not isinstance(v, list)})

        # Plot Confusion Matrix for Quantized CPU
        if 'true_labels' in st.session_state.quantized_metrics:
            st.write("#### Quantized CPU Confusion Matrix")
            cm_fig_quant = training_script.generate_confusion_matrix_plot(
                st.session_state.quantized_metrics['true_labels'],
                st.session_state.quantized_metrics['pred_labels']
            )
            st.pyplot(cm_fig_quant)
            cm_fig_quant.savefig(os.path.join(st.session_state.report_dir, 'quantized_cpu_confusion_matrix.png'))
    else:
        st.error("Could not run quantization test because the trained CPU model object was not available.")


    # --- GPU Training ---
    if training_script.is_gpu_available():
        st.write("### 🚀 Training on GPU...")
        gpu_chart_placeholder = st.empty()
        train_progress_bar_gpu = st.progress(0)
        train_status_text_gpu = st.empty()
        def gpu_train_progress_callback(progress, text):
            train_progress_bar_gpu.progress(progress)
            train_status_text_gpu.text(text)

        st.write("### 🧪 Testing on GPU...")
        test_progress_bar_gpu = st.progress(0)
        test_status_text_gpu = st.empty()
        def gpu_test_progress_callback(progress, text):
            test_progress_bar_gpu.progress(progress)
            test_status_text_gpu.text(text)

        st.session_state.gpu_metrics = training_script.run_gpu_training(
            learning_rate, num_epochs, batch_size, hidden_layers, hidden_size,
            train_percentage=train_percentage,
            dataset_name=dataset_name,
            save_model=save_models_checkbox,
            train_progress_callback=gpu_train_progress_callback,
            test_progress_callback=gpu_test_progress_callback
        )
        train_status_text_gpu.text("✅ GPU training complete!")
        test_status_text_gpu.text("✅ GPU testing complete!")
        st.success("✅ GPU process complete!")
        st.write("### 🚀 GPU Performance Metrics")
        st.json({k: v for k, v in st.session_state.gpu_metrics.items() if not isinstance(v, list)})

        # Plot GPU history
        if 'train_loss_history' in st.session_state.gpu_metrics:
            st.write("#### GPU Training History")
            gpu_fig = plot_history(st.session_state.gpu_metrics, "GPU")
            gpu_chart_placeholder.pyplot(gpu_fig)
            gpu_fig.savefig(os.path.join(st.session_state.report_dir, 'gpu_training_history.png'))
        
        # Plot GPU utilization
        if 'cpu_util_history' in st.session_state.gpu_metrics:
            st.write("#### GPU Resource Utilization")
            gpu_util_fig = plot_utilization_history(st.session_state.gpu_metrics, "GPU")
            st.pyplot(gpu_util_fig)
            gpu_util_fig.savefig(os.path.join(st.session_state.report_dir, 'gpu_resource_utilization.png'))

        # Plot Confusion Matrix for GPU
        if 'true_labels' in st.session_state.gpu_metrics:
            st.write("#### GPU Confusion Matrix")
            cm_fig_gpu = training_script.generate_confusion_matrix_plot(
                st.session_state.gpu_metrics['true_labels'],
                st.session_state.gpu_metrics['pred_labels']
            )
            st.pyplot(cm_fig_gpu)
            cm_fig_gpu.savefig(os.path.join(st.session_state.report_dir, 'gpu_confusion_matrix.png'))

        # --- Comparison ---
        st.write("### 📈 Comparison")
        st.session_state.comparison_df, st.session_state.chart_path = training_script.generate_comparison(
            st.session_state.cpu_metrics, 
            st.session_state.gpu_metrics,
            st.session_state.quantized_metrics,
            report_dir=st.session_state.report_dir
        )
        st.session_state.comparison_df.to_csv(os.path.join(st.session_state.report_dir, 'comparison_metrics.csv'))
    else:
        st.warning("GPU not available. Skipping GPU training and comparison.")

if st.session_state.comparison_df is not None:
    st.write("### 📋 Detailed Comparison Table")
    st.dataframe(st.session_state.comparison_df, use_container_width=True)

if st.session_state.chart_path is not None:
    st.write("### 📊 Performance Chart")
    st.image(st.session_state.chart_path, caption="CPU vs. GPU Performance")

st.write("---")
st.header("💾 Load and Test a Saved Model")

model_to_load = st.selectbox("Select Model to Load", ["cpu_model.pth", "gpu_model.pth"])
device_to_use = st.selectbox("Select Device for Testing", ["CPU", "GPU"])

if st.button("🧪 Test Saved Model"):
    device_type = 'cpu' if device_to_use == 'CPU' else 'gpu'
    
    st.write(f"### 🧪 Testing {model_to_load} on {device_to_use}...")
    test_progress_bar_load = st.progress(0)
    test_status_text_load = st.empty()
    def load_test_progress_callback(progress, text):
        test_progress_bar_load.progress(progress)
        test_status_text_load.text(text)

    loaded_metrics = training_script.load_and_test_model(
        model_path=model_to_load,
        device_type=device_type,
        batch_size=batch_size,
        hidden_layers=hidden_layers,
        hidden_size=hidden_size,
        test_progress_callback=load_test_progress_callback
    )
    
    if "Error" in loaded_metrics:
        st.error(loaded_metrics["Error"])
    else:
        st.success(f"✅ Testing complete for {model_to_load}!")
        st.write("### 📊 Loaded Model Performance")
        st.json({k: v for k, v in loaded_metrics.items() if not isinstance(v, list)})

        # Plot Confusion Matrix for loaded model
        if 'true_labels' in loaded_metrics:
            st.write("#### Loaded Model Confusion Matrix")
            cm_fig_load = training_script.generate_confusion_matrix_plot(
                loaded_metrics['true_labels'],
                loaded_metrics['pred_labels']
            )
            st.pyplot(cm_fig_load)
