# 🐑 CapraMind-V1-small  
**A Specialized Large Language Model for Dairy Goat Farming Management**

> **CapraMind-V1-small** is a lightweight, domain-specific language model fine-tuned on high-quality dairy goat husbandry knowledge. It provides accurate, safe, and practical guidance on nutrition, reproduction, disease prevention, milking, and daily management—tailored for farmers, veterinarians, and agricultural extension workers.

---

## 📌 Overview

- **Base Model**: Qwen3-1.7B-Base  
- **Finetuning Method**: LoRA (Low-Rank Adaptation)  
- **Training Data**: 2,755+ expert-verified QA pairs from veterinary manuals, FAO guidelines, and Chinese agricultural standards  
- **Parameters**: ~1.7B (LoRA rank=8, alpha=16)  
- **Languages**: Simplified Chinese (primary), with technical terms in English (e.g., "glucose calcium", "mastitis")  
- **License**: Non-commercial use only (see [License](#-license))

---

## ✨ Key Features

✅ **Expert-Level Knowledge**  
Covers core topics:  
- Estrus detection & breeding management  
- Lactation nutrition & feed formulation  
- Common diseases (e.g., mastitis, postpartum hypocalcemia)  
- Vaccination & deworming schedules  
- Milking hygiene & udder care  

✅ **Safety-First Design**  
- Refuses to recommend human medications or unverified treatments  
- Cites standard protocols (e.g., “According to the *Dairy Goat Management Handbook*, 2022…”)  
- Falls back to: “Please consult a licensed veterinarian” for high-risk queries  

✅ **Lightweight & Deployable**  
- Runs on consumer GPUs (e.g., RTX 3090/4090 with 24GB VRAM)  
- Supports GGUF quantization for CPU/edge deployment  

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install transformers accelerate torch sentencepiece
```

### 2. Load the Model (Hugging Face Style)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "CapraMind-V1-small"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto"
)
model.eval()
```

### 3. Inference Example
```python
prompt = "奶山羊产后瘫痪的紧急处理措施是什么？"
messages = [
    {"role": "system", "content": "你是一名资深奶山羊兽医，请基于专业规范回答。"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

> 💡 **Note**: Use the **Qwen chat template** (`tokenizer.apply_chat_template`) for best results.

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Final Training Loss | 1.55 |
| Validation Loss | ~1.58 (on held-out 10% data) |
| Accuracy (Expert Eval, n=50) | 92% (factual correctness) |
| Hallucination Rate | < 4% (mostly on rare diseases) |

> Evaluated by 2 certified livestock veterinarians on real-world farmer questions.

---

## 📁 Repository Structure

```
CapraMind-V1-small/
├── adapter_config.json    # LoRA config (rank=8, alpha=16)
├── adapter_model.safetensors  # LoRA weights
├── README.md
├── training_args.json     # Full training hyperparameters
└── eval_samples.json      # Sample Q&A for validation
```

---

## ⚠️ Limitations

- **Not a substitute for veterinary care**: For emergencies, always contact a professional.
- **Coverage bias**: Strong in common diseases & nutrition; weaker in rare genetic disorders.
- **Language**: Optimized for Chinese-speaking users; English queries may yield suboptimal results.

---

## 🔒 License

This model is released under the **[MIT License](https://opensource.org/licenses/MIT)**.

- ✅ You may freely use, modify, and distribute this model for **both non-commercial and commercial purposes**.
- ✅ You are required to include the original copyright notice and license text in any redistribution.

> ⚠️ **Important Note**:  
> This model is built upon **Qwen3-1.7B-Base**, which is licensed under Alibaba’s **[Qwen License](https://huggingface.co/Qwen/Qwen3-1.7B-Base)**.  
> The MIT license applies **only to the LoRA adapter weights and training code** in this repository.  
> **You must comply with Qwen3’s license terms when using or redistributing the base model.**  
>  
> In practice:  
> - You may deploy CapraMind-V1-small in your app or service.  
> - But you must not redistribute Qwen3-1.7B-Base unless you comply with Alibaba’s terms (e.g., attribution, non-resale of base weights).  
>  
> For commercial deployment, we recommend using **only the LoRA adapter** alongside a legally obtained Qwen3 base model.

---

## 🙏 Acknowledgements

- Training data curated from:  
  - *Dairy Goat Production Handbook* (China Agricultural Press, 2022)  
  - FAO Guidelines on Small Ruminant Management  
  - National Dairy Goat Association (NDGA) protocols  
- Built with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [Qwen](https://huggingface.co/Qwen)
