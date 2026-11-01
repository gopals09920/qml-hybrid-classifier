import streamlit as st
import torch
import torch.nn as nn
import plotly.graph_objects as go
from model import QMLHybridClassifier

# Page Configuration
st.set_page_config(
    page_title="Quantum ML Dashboard",
    page_icon="⚛️",
    layout="wide"
)

# Custom CSS for Futuristic & Sci-Fi Aesthetic
st.markdown("""
<style>
    .main {
        background-color: #050505;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%);
        color: #000;
        font-weight: 800;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        box-shadow: 0px 0px 20px rgba(0, 242, 254, 0.6);
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0px 0px 30px rgba(79, 172, 254, 0.9);
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<h1 style='text-align: center; color: #00F2FE; text-shadow: 0px 0px 15px rgba(0,242,254,0.5);'>⚛️ QUANTUM NEURAL HYBRID INTERFACE 🚀</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0AEC0;'>Parameterized Quantum Circuits (PQCs) with PyTorch Autograd Integration </p>", unsafe_allow_html=True)
st.divider()

# Sidebar Controls
st.sidebar.header("🎛️ Qubit Parameters")
st.sidebar.markdown("Fine-tune quantum state vectors:")

f1 = st.sidebar.slider("Feature 1 (Amplitude State)", -3.0, 3.0, 0.0, 0.1)
f2 = st.sidebar.slider("Feature 2 (Phase Angle)", -3.0, 3.0, 0.0, 0.1)
f3 = st.sidebar.slider("Feature 3 (Rotation Gate)", -3.0, 3.0, 0.0, 0.1)
f4 = st.sidebar.slider("Feature 4 (Entanglement)", -3.0, 3.0, 0.0, 0.1)

# Load Model
@st.cache_resource
def load_model():
    model = QMLHybridClassifier(n_features=4, n_qubits=4, n_classes=2)
    model.eval()
    return model

model = load_model()

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📡 Live Quantum State Vector")
    st.code(f"|ψ⟩ = AngleEmbedding([{f1}, {f2}, {f3}, {f4}])", language="python")
    
    st.markdown("### 🔬 Hardware Status")
    st.success("🟢 Simulated QNodes Online | Backend: default.qubit")
    
with col2:
    st.subheader("⚡ Quantum Execution Engine")
    if st.button("🔥 Execute Quantum Circuit Inference", use_container_width=True):
        with st.spinner("Computing Superposition & Backpropagation..."):
            input_tensor = torch.tensor([[f1, f2, f3, f4]], dtype=torch.float32)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = output[0].tolist()
                pred_class = torch.argmax(output, dim=1).item()
                confidence = probs[pred_class] * 100

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Metrics Display
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="🎯 Predicted State", value=f"Class {pred_class}")
        with m_col2:
            st.metric(label="📈 Confidence Score", value=f"{confidence:.2f}%")
            
        st.markdown("### 📊 Interactive Probability Spectrum")
        
        # Plotly Fancy Interactive Graph
        classes = ['Class 0', 'Class 1']
        probabilities = [probs[0] * 100, probs[1] * 100]
        
        fig = go.Figure(data=[
            go.Bar(
                x=classes,
                y=probabilities,
                marker=dict(
                    color=['#FF007F', '#00F2FE'],
                    line=dict(color='#FFFFFF', width=1.5)
                ),
                text=[f"{p:.2f}%" for p in probabilities],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            paper_bgcolor='#050505',
            plot_bgcolor='#0D1117',
            font=dict(color='white', family='Courier New'),
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis=dict(title='Probability (%)', range=[0, 100], gridcolor='#30363d'),
            xaxis=dict(gridcolor='#30363d'),
            height=280
        )
        
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("<p style='text-align: center; color: #718096;'>Powered by PennyLane, PyTorch & Plotly • Next-Gen QML Demo</p>", unsafe_allow_html=True)