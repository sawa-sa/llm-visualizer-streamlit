import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st

@st.cache_resource
def load_model(model_name: str = "gpt2-medium"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model.to(device), tokenizer, device