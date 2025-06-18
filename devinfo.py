import torch
import streamlit as st

def show_device_info():
    st.sidebar.markdown("## 💻 実行環境")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.write(f"使用デバイス: `{device}`")
    if device.type == "cuda":
        st.sidebar.write(f"GPU 名称: `{torch.cuda.get_device_name(0)}`")
        st.sidebar.write(f"メモリ使用量: {torch.cuda.memory_allocated() / 1024**2:.1f} MiB")
    return device
