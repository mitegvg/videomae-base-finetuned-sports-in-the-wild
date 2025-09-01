# Contextual Bridge Optimizer (CBO)

The **Contextual Bridge Optimizer (CBO)** is a lightweight service designed to make the **TMAD (Tri-Model Asymmetric Distillation)** framework easier to use in research workflows.  

It automates:
- **Student architecture selection** (hidden size, depth, heads, MLP ratio, etc.)  
- **Distillation weights** (classification, feature, attention, logits, asymmetric)  
- **Teacher/Assistant ratio (bridge policy)** based on dataset similarity and entropy  
- **Training configuration** (Hugging Face `TrainingArguments`)  

All from **simple user requirements** such as model size range, target accuracy, and compute constraints.  

The output is two configs:
1. **Assistant config** (TMAD distillation + bridge settings)  
2. **Student config** (architecture + training config, ready for Transformers)  

---

## 🚀 Features
- Embeddable directly in **Jupyter / Colab notebooks**  
- Scans datasets to infer **domain embeddings, label structure, class count, clip stats**  
- User provides **requirements** (e.g., *20–50 MB student model with ≥80% accuracy*)  
- Generates **assistant + student YAML configs** and **Transformers training args**  
- Simple **UI form** with Optimize button  

---

## 📦 Installation

```bash
pip install cbo-tmad