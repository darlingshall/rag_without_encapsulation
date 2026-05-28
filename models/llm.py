import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from config import LOCAL_LLM_DIR, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY

def create_llm():
    print("Loading Phi-3 model...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_DIR, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_LLM_DIR,
        device_map={"": "cpu"},
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model
