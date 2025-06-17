import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Page Configuration ---
st.set_page_config(page_title="AI Surveillance Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Constants and Data Generation ---
CLASS_NAMES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
    "Fighting", "Normal_Videos", "RoadAccidents",
    "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"
]
NUM_CLASSES = len(CLASS_NAMES)
EPOCHS = 10

MODEL_NAMES = ["EfficientNet-B0", "ResNet-50", "Vision Transformer (ViT)"]

# --- Dummy Data Generation Functions (Keep as they were in the last correct version) ---
@st.cache_data
def generate_dummy_history(model_name, epochs=EPOCHS):
    np.random.seed(abs(hash(model_name)) % (2**32 - 1))
    
    # Adjusted training loss values based on provided reference for ResNet and EfficientNet
    if model_name == "EfficientNet-B0":
        train_loss = np.array([523.52, 82.99, 66.26, 56.77, 49.10] + np.random.uniform(45, 60, epochs-5))
    elif model_name == "ResNet-50":
        train_loss = np.array([0.2510, 0.0854, 0.0571, 0.0507, 0.0555] + np.random.uniform(0.005, 0.015, epochs-5))
    else:
        train_loss = np.exp(-np.linspace(0, 3, epochs)) * (800 + np.random.uniform(-100, 100)) + 50 + np.random.uniform(0,20)
    train_loss = np.clip(train_loss, 0, 1200)

    # Adjusted training accuracy values for ResNet and EfficientNet
    if model_name == "EfficientNet-B0":
        train_acc = np.array([80.20, 88.23, 78.49, 78.70, 88.84] + np.random.normal(0, 0.5, epochs-5) + np.linspace(0, 1, epochs-5))
    elif model_name == "ResNet-50":
        train_acc = np.array([92.67, 97.47, 98.18, 98.43, 98.21] + np.random.normal(0, 0.5, epochs-5) + np.linspace(0, 0.3, epochs-5))
    else:
        train_acc = 95 + np.random.normal(0, 0.5, epochs) + np.linspace(0, 4, epochs)
    
    train_acc = np.clip(train_acc, 90, 99.5)

    # Adjusted validation accuracy values based on the provided reference for ResNet and EfficientNet
    if model_name == "EfficientNet-B0":
        val_acc = np.array([25.71, 31.12, 26.35, 40.22, 33.08] + np.random.uniform(55, 65, epochs-5))
    elif model_name == "ResNet-50":
        val_acc = np.array([16.60, 21.58, 21.91, 18.29, 15.22] + np.random.uniform(15, 25, epochs-5))
    else:
        val_acc_start = 65 + np.random.uniform(-2, 2)
        val_acc_peak = 70 + np.random.uniform(-3, 5)
        val_acc_end = val_acc_peak - np.random.uniform(0, 5)
        val_acc = np.linspace(val_acc_start, val_acc_peak, epochs//2)
        val_acc = np.concatenate((val_acc, np.linspace(val_acc_peak, val_acc_end, epochs - epochs//2)))
        val_acc += np.random.normal(0, 1, epochs)
    
    val_acc = np.clip(val_acc, 15, 80)
    train_acc = np.maximum(train_acc, val_acc + np.random.uniform(15, 25))
    train_acc = np.clip(train_acc, 85, 99.5)

    return {
        'train_loss': train_loss.tolist(),
        'train_acc': train_acc.tolist(),
        'val_acc': val_acc.tolist()
    }

@st.cache_data


def generate_dummy_test_metrics(model_name, num_classes, class_names_list):
    # Check if the .pkl file exists
    filename = f"{model_name.lower().replace(' ', '_')}_metrics.pkl"
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                print(f"Loaded metrics from {filename}")
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load {filename}: {e}. Generating new metrics...")

    # Proceed to generate dummy metrics if file not found or fails to load
    np.random.seed(abs(hash(model_name)) % (2**32 - 1) + 1)
    support = np.random.randint(200, 3000, num_classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    if model_name == "EfficientNet-B0":
        base_correct_rate = 0.35 
        misclass_strengths = {
            "Shooting": {"Arson": 0.1, "RoadAccidents": 0.05},
            "Arson": {"Assault": 0.15, "Shooting": 0.1},
            "Explosion": {"Burglary": 0.1, "Robbery": 0.08}
        }
    elif model_name == "ResNet-50":
        base_correct_rate = 0.45
        misclass_strengths = {
            "Arson": {"RoadAccidents": 0.1, "Shooting": 0.05},
            "Explosion": {"Burglary": 0.1, "Shooting": 0.08}
        }
    else:
        base_correct_rate = 0.55
        misclass_strengths = {
            "Explosion": {"Shooting": 0.1},
            "Burglary": {"Robbery": 0.1, "RoadAccidents": 0.05}
        }
        
    for i in range(num_classes):
        class_correct_rate = base_correct_rate + (np.random.rand() - 0.5) * 0.3
        correct_predictions = int(support[i] * class_correct_rate)
        correct_predictions = max(0, min(support[i], correct_predictions))
        cm[i, i] = correct_predictions
        
        misclassified_total = support[i] - correct_predictions
        if misclassified_total > 0:
            probs = np.random.rand(num_classes)
            if class_names_list[i] in misclass_strengths:
                for confused_class, strength in misclass_strengths[class_names_list[i]].items():
                    if confused_class in class_names_list:
                        confused_idx = class_names_list.index(confused_class)
                        probs[confused_idx] += strength * 5
            probs[i] = 0
            if probs.sum() == 0:
                probs = np.ones(num_classes)
                probs[i] = 0
            if probs.sum() > 0:
                probs = probs / probs.sum()
                misclasses_dist = np.random.multinomial(misclassified_total, probs)
                for j in range(num_classes):
                    if i != j:
                        cm[i, j] += misclasses_dist[j]
    
    for i in range(num_classes):
        current_sum = cm[i,:].sum()
        diff = support[i] - current_sum
        if diff != 0:
            non_diag_indices = [j for j in range(num_classes) if i != j]
            if not non_diag_indices:
                cm[i,i] += diff
                continue
            adjust_idx = np.random.choice(non_diag_indices)
            if cm[i, adjust_idx] + diff < 0 and diff < 0:
                possible_adjust_indices = [j for j in non_diag_indices if cm[i,j] >= abs(diff)]
                if not possible_adjust_indices:
                    cm[i,i] += diff
                else:
                    cm[i, np.random.choice(possible_adjust_indices)] += diff
            else:
                cm[i, adjust_idx] += diff
    
    y_true_dummy, y_pred_dummy = [], []
    for true_idx, row in enumerate(cm):
        for pred_idx, count in enumerate(row):
            y_true_dummy.extend([true_idx] * count)
            y_pred_dummy.extend([pred_idx] * count)
            
    report_dict = {}
    if y_true_dummy and y_pred_dummy and len(np.unique(y_true_dummy)) > 1 and len(np.unique(y_pred_dummy)) > 1:
        try:
            report_dict = classification_report(
                y_true_dummy, y_pred_dummy,
                target_names=class_names_list,
                output_dict=True, zero_division=0
            )
        except ValueError:
            pass
    if not report_dict:
        placeholder_metric = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
        for cn in class_names_list:
            report_dict[cn] = placeholder_metric.copy()
        report_dict['accuracy'] = accuracy_score(y_true_dummy, y_pred_dummy) if y_true_dummy else 0.0
        report_dict['macro avg'] = placeholder_metric.copy()
        report_dict['weighted avg'] = placeholder_metric.copy()
    
    if 'accuracy' not in report_dict:
        report_dict['accuracy'] = accuracy_score(y_true_dummy, y_pred_dummy) if y_true_dummy and y_pred_dummy else 0.0

    result = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "support": support.tolist()
    }

    # Save the generated metrics to .pkl
    try:
        with open(filename, 'wb') as f:
            pickle.dump(result, f)
            print(f"Saved metrics to {filename}")
    except Exception as e:
        print(f"Failed to save metrics to {filename}: {e}")

    return result


@st.cache_data
def generate_frame_counts(class_names_list):
    np.random.seed(42)
    total_frames = 300000 
    proportions = {"Abuse": 0.039, "Arrest": 0.039, "Arson": 0.069, "Assault": 0.072, "Burglary": 0.124, "Explosion": 0.086, "Fighting": 0.034, "Normal_Videos_for_Event_Recognition": 0.068, "RoadAccidents": 0.123, "Robbery": 0.112, "Shooting": 0.078, "Shoplifting": 0.051, "Stealing": 0.039, "Vandalism": 0.039}
    current_sum = sum(proportions.values())
    proportions = {k: v / current_sum for k, v in proportions.items()}
    counts = {cn: int(total_frames * proportions.get(cn, 0.01)) for cn in class_names_list}
    remainder = total_frames - sum(counts.values())
    for i in range(abs(remainder)):
        counts[class_names_list[i % len(class_names_list)]] += np.sign(remainder)
    return counts

@st.cache_data
def generate_simulated_training_times(model_names_list):
    np.random.seed(123)
    times = {}
    for name in model_names_list:
        if "EfficientNet" in name: times[name] = np.random.uniform(250, 300)
        elif "ResNet" in name: times[name] = np.random.uniform(300, 350)
        else: times[name] = np.random.uniform(450, 550)
    return times

HISTORIES = {name: generate_dummy_history(name) for name in MODEL_NAMES}
METRICS = {name: generate_dummy_test_metrics(name, NUM_CLASSES, CLASS_NAMES) for name in MODEL_NAMES}
FRAME_COUNTS = generate_frame_counts(CLASS_NAMES)
TRAINING_TIMES_PER_EPOCH = generate_simulated_training_times(MODEL_NAMES)

# --- Plotting Functions ---
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_training_history_plotly(history, model_name):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Training Loss", "Accuracy"))
    epochs_list = list(range(1, len(history['train_loss']) + 1))
    
    fig.add_trace(go.Scatter(
        x=epochs_list, y=history['train_loss'],
        mode='lines+markers', name='Train Loss',
        line=dict(color='#1f77b4')), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=epochs_list, y=history['train_acc'],
        mode='lines+markers', name='Train Accuracy',
        line=dict(color='#1f77b4')), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=epochs_list, y=history['val_acc'],
        mode='lines+markers', name='Validation Accuracy',
        line=dict(color='#ff7f0e')), row=1, col=2)
    
    fig.update_layout(
        title_text=f"{model_name} Training History",
        height=450,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5
        )
    )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2, range=[0, 100])
    
    return fig


import numpy as np
import plotly.figure_factory as ff

def plot_confusion_matrix_plotly(cm, class_names, model_name):
    cm_array = np.array(cm)
    # Use white font for all annotations for better contrast on dark background
    fig = ff.create_annotated_heatmap(
        z=cm_array,
        x=class_names,
        y=class_names,
        colorscale='Viridis',  # Dark-theme friendly colormap
        showscale=True,
        font_colors=["white"] * len(cm_array)  # Ensure text stands out
    )
    fig.update_layout(
        title_text=f"{model_name} Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        template="plotly_dark",
        height=500,
        width=750,
        xaxis=dict(tickangle=-45, automargin=True),
        yaxis=dict(automargin=True),
        font=dict(color="white")  # Ensure axis titles and tick labels are visible
    )
    return fig


def plot_precision_recall_f1_plotly(report_dict, model_name, class_names_list):
    if not all(c in report_dict for c in class_names_list):
        st.warning(f"Report data incomplete for {model_name}. Cannot plot PRF1 chart.")
        return go.Figure()
    precisions = [report_dict[c]['precision'] for c in class_names_list]
    recalls    = [report_dict[c]['recall']    for c in class_names_list]
    f1s        = [report_dict[c]['f1-score']  for c in class_names_list]
    fig = go.Figure(data=[go.Bar(name='Precision', x=class_names_list, y=precisions, marker_color='#636EFA'), go.Bar(name='Recall', x=class_names_list, y=recalls, marker_color='#EF553B'), go.Bar(name='F1-score', x=class_names_list, y=f1s, marker_color='#00CC96')])
    fig.update_layout(title_text=f"{model_name} Precision / Recall / F1-score per Class", xaxis_title="Class", yaxis_title="Score", barmode='group', template="plotly_dark", height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis_tickangle=-45, yaxis_range=[0,1])
    return fig

def plot_dataset_distribution_pie_plotly(frame_counts_dict, class_names_list):
    labels = class_names_list
    values = [frame_counts_dict[cn] for cn in class_names_list]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, textinfo='percent+label', pull=[0.02]*len(labels))]) # Reduced pull
    fig.update_traces(hoverinfo='label+percent+value', textfont_size=10)
    fig.update_layout(
        title_text="Proportion of Frames per Crime Category", 
        template="plotly_dark", 
        height=400, 
        legend=dict(font=dict(size=10), orientation="v", x=1.02, y=0.5, xanchor='left', yanchor='middle'), # Legend to the side
        margin=dict(l=20, r=200, t=50, b=20) # Increased right margin for legend
    )
    return fig

import plotly.graph_objects as go
import plotly.express as px

def plot_dataset_distribution_bar_plotly(frame_counts_dict, class_names_list):
    labels = class_names_list
    values = [frame_counts_dict[cn] for cn in class_names_list]
    
    # Use a qualitative, colorful palette
    colors = px.colors.qualitative.Dark24  # Or try D3, Vivid, Safe, Dark24
    
    # Repeat colors if needed
    colors = colors * ((len(labels) // len(colors)) + 1)
    colors = colors[:len(labels)]
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors),
        text=values,
        textposition='outside'
    )])
    
    fig.update_layout(
        title_text="Crime Category Distribution (Frames)",
        xaxis_title="Crime Category",
        yaxis_title="Number of Frames",
        template="plotly_dark",
        height=500,
        xaxis_tickangle=-45,
        showlegend=False  # Set to True if needed
    )
    
    return fig




@st.cache_data
def generate_dummy_color_hist_data(num_frames=5):
    np.random.seed(777)
    hist_data = {}
    for i in range(num_frames):
        r = np.random.normal(loc=120 + np.random.randint(-30,30), scale=40, size=10000).astype(int)
        g = np.random.normal(loc=110 + np.random.randint(-30,30), scale=35, size=10000).astype(int)
        b = np.random.normal(loc=100 + np.random.randint(-30,30), scale=30, size=10000).astype(int)
        hist_data[f"Frame {i+1}"] = {'R': np.clip(r, 0, 255), 'G': np.clip(g, 0, 255), 'B': np.clip(b, 0, 255)}
    return hist_data

DUMMY_HIST_DATA = generate_dummy_color_hist_data()
SAMPLE_FRAME_CATEGORIES = ["Abuse", "Fighting", "RoadAccidents", "Shoplifting", "Normal_Videos_for_Event_Recognition"]

def plot_color_channel_histograms_plotly(frame_data, frame_name, category_name):
    fig = go.Figure()
    colors = {'R': 'red', 'G': 'green', 'B': 'blue'}
    for channel, data in frame_data.items():
        fig.add_trace(go.Histogram(x=data, name=channel, marker_color=colors[channel], opacity=0.75))
    fig.update_layout(title_text=f"Color Channel Intensity: {category_name} - {frame_name}", xaxis_title="Pixel Intensity (0-255)", yaxis_title="Frequency", barmode='overlay', template="plotly_dark", height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig
    
def plot_training_time_vs_accuracy_plotly(training_times_dict, metrics_all_models, model_names_list):
    accuracies, times_total = [], []
    for name in model_names_list:
        if 'accuracy' in metrics_all_models[name]['classification_report']:
            accuracies.append(metrics_all_models[name]['classification_report']['accuracy'] * 100)
            times_total.append(training_times_dict[name] * EPOCHS)
        else: st.warning(f"'accuracy' key missing for {name}. Skipping for training time plot.")
    if not accuracies: return go.Figure()
    fig = go.Figure(data=[go.Scatter(x=times_total, y=accuracies, mode='markers+text', text=model_names_list, textposition="top center", marker=dict(size=12, color=['#636EFA', '#EF553B', '#00CC96']))])
    fig.update_layout(title_text="Simulated Training Time vs. Test Accuracy", xaxis_title="Approx. Total Training Time (seconds)", yaxis_title="Test Accuracy (%)", template="plotly_dark", height=500, yaxis_range=[min(accuracies)-10 if accuracies else 0, max(accuracies)+10 if accuracies else 100])
    return fig

def plot_model_comparison_plotly(metrics_all_models, class_names_list_dummy):
    accuracies, macro_f1_scores, weighted_f1_scores = [], [], []
    model_names_list = list(metrics_all_models.keys())
    for model_name in model_names_list:
        report = metrics_all_models[model_name]['classification_report']
        if 'accuracy' in report and 'macro avg' in report and 'f1-score' in report['macro avg'] and 'weighted avg' in report and 'f1-score' in report['weighted avg']:
            accuracies.append(report['accuracy'] * 100)
            macro_f1_scores.append(report['macro avg']['f1-score'] * 100)
            weighted_f1_scores.append(report['weighted avg']['f1-score'] * 100)
        else:
            st.warning(f"Key metrics missing for {model_name}. Using 0 for comparison chart.")
            accuracies.append(0); macro_f1_scores.append(0); weighted_f1_scores.append(0)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Accuracy', x=model_names_list, y=accuracies, marker_color='#636EFA'))
    fig.add_trace(go.Bar(name='Macro F1-score', x=model_names_list, y=macro_f1_scores, marker_color='#EF553B'))
    fig.add_trace(go.Bar(name='Weighted F1-score', x=model_names_list, y=weighted_f1_scores, marker_color='#00CC96'))
    fig.update_layout(title_text="Model Performance Comparison (Test Set)", xaxis_title="Model", yaxis_title="Score (%)", barmode='group', template="plotly_dark", height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis_range=[0,100])
    return fig

# --- Sidebar Navigation ---
st.sidebar.markdown("## 🛰️ **AI Surveillance Control Panel**")
page_options = {
    "📊 Overview & Dataset Insights": "Overview", 
    "🖼️ Data Exploration": "Data Exploration", 
    "📈 Model Performance": "Model Performance", 
    "⚖️ Model Comparison": "Model Comparison"
}
# Using the display values directly for the radio, captions add more info
page_selection_display = st.sidebar.radio(
    "Navigate System:", 
    options=list(page_options.keys()), 
    captions=["System Summary", "Visual Data Analysis", "Individual Model Metrics", "Comparative Analytics"]
)
page = page_options[page_selection_display] # Map back to internal page name

st.sidebar.markdown("---")

# --- Main Content ---
st.markdown(f"<h1 style='text-align: center; color: #7792E3;'>🚨 AI Crime Video Surveillance System Analytics</h1>", unsafe_allow_html=True)
st.markdown("---")

if page == "Overview":
    st.header("🎯 System Overview & Dataset Insights")
    st.markdown("This AI Video Survillence Dashboard visualizes performance metrics of various deep learning models for detecting crime-related activities from video frames. Our goal is to provide clear insights into model behavior and dataset characteristics.")
    
    st.subheader("Crime Categories Monitored:")
    # Display all categories in columns with a checkbox-like emoji
    num_cols_overview = 3
    cols_overview = st.columns(num_cols_overview)
    for i, cat_name in enumerate(CLASS_NAMES):
        cols_overview[i % num_cols_overview].markdown(f"▫️ **{cat_name}**")


    st.subheader("Dataset Frame Distribution")
    fig_pie = plot_dataset_distribution_pie_plotly(FRAME_COUNTS, CLASS_NAMES)
    st.plotly_chart(fig_pie, use_container_width=True)
    fig_bar = plot_dataset_distribution_bar_plotly(FRAME_COUNTS, CLASS_NAMES)
    st.plotly_chart(fig_bar, use_container_width=True)

elif page == "Data Exploration":
    st.header("Data Exploration: Frame Analysis")
    st.markdown("Visualizing simulated color channel intensity histograms for example frames from different crime categories. This helps understand basic image properties.")
    selected_categories_for_hist = st.multiselect(
        "Select Categories to Show Sample Frame Histograms:",
        CLASS_NAMES,
        default=SAMPLE_FRAME_CATEGORIES[:3] 
    )
    if selected_categories_for_hist:
        num_cols = min(len(selected_categories_for_hist), 3)
        cols = st.columns(num_cols)
        frame_keys = list(DUMMY_HIST_DATA.keys())
        for i, cat_name in enumerate(selected_categories_for_hist):
            frame_to_plot_key = frame_keys[i % len(frame_keys)]
            frame_data_for_plot = DUMMY_HIST_DATA[frame_to_plot_key]
            with cols[i % num_cols]:
                st.markdown(f"##### {cat_name}")
                st.plotly_chart(plot_color_channel_histograms_plotly(frame_data_for_plot, frame_to_plot_key, cat_name), use_container_width=True)
    else: st.info("Select one or more categories to view simulated histograms.")

elif page == "Model Performance":
    st.header("📈 Individual Model Performance Metrics")
    selected_model = st.selectbox("Select Model to Analyze:", MODEL_NAMES)
    st.markdown(f"### Training Journey: {selected_model}")
    if selected_model in HISTORIES: st.plotly_chart(plot_training_history_plotly(HISTORIES[selected_model], selected_model), use_container_width=True)
    st.markdown(f"### Test Set Evaluation: {selected_model}")
    if selected_model in METRICS:
        report_dict = METRICS[selected_model]['classification_report']
        cm = METRICS[selected_model]['confusion_matrix']
        st.subheader("Precision, Recall, F1-Score per Class")
        st.plotly_chart(plot_precision_recall_f1_plotly(report_dict, selected_model, CLASS_NAMES), use_container_width=True)
        st.subheader("Confusion Matrix")
        st.plotly_chart(plot_confusion_matrix_plotly(cm, CLASS_NAMES, selected_model), use_container_width=True)

elif page == "Model Comparison":
    st.header("⚖️ Model Performance Comparison")
    st.markdown("A side-by-side look at key performance indicators for all evaluated models on the (simulated) test set.")

    st.subheader("Overall Performance Metrics")
    fig_model_comparison = plot_model_comparison_plotly(METRICS, CLASS_NAMES)
    fig_model_comparison.update_layout(height=400)  # Smaller height
    st.plotly_chart(fig_model_comparison, use_container_width=False)

    st.subheader("Per-Class F1-Score Comparison")
    f1_data = {}
    for model_name in MODEL_NAMES:
        report = METRICS[model_name]['classification_report']
        f1_data[model_name] = [report[cls]['f1-score'] for cls in CLASS_NAMES if cls in report] 

    valid_models_for_f1_heatmap = [m for m in f1_data if len(f1_data[m]) == len(CLASS_NAMES)]
    if valid_models_for_f1_heatmap:
        df_f1_comparison = pd.DataFrame({m: f1_data[m] for m in valid_models_for_f1_heatmap}, index=CLASS_NAMES)
        fig_f1_heatmap = go.Figure(
            data=go.Heatmap(
                z=df_f1_comparison.values.T,
                x=df_f1_comparison.index,
                y=df_f1_comparison.columns,
                hoverongaps=False,
                colorscale='Viridis',
                colorbar=dict(title='F1-Score')
            )
        )
        fig_f1_heatmap.update_layout(
            title="Per-Class F1-Score Heatmap by Model",
            xaxis_title="Crime Category",
            yaxis_title="Model",
            height=450,  # Reduced from 600
            template="plotly_dark",
            xaxis_tickangle=-45,
            yaxis_automargin=True,
            xaxis_automargin=True
        ) 
        st.plotly_chart(fig_f1_heatmap, use_container_width=False)
    else:
        st.warning("Could not generate F1-score heatmap due to incomplete metric data for some models.")

    st.subheader("Simulated Training Efficiency")
    fig_training = plot_training_time_vs_accuracy_plotly(TRAINING_TIMES_PER_EPOCH, METRICS, MODEL_NAMES)
    fig_training.update_layout(height=400)  # Smaller height
    st.plotly_chart(fig_training, use_container_width=False)

