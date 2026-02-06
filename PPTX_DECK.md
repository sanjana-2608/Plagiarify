---
marp: true
paginate: true
theme: default
size: 16:9
---

# Semantic Plagiarism Detection
### Sentence Embeddings + Cosine Similarity
**Hackathon Project**

**Team:** [Team Name]
**Members:** [Name 1], [Name 2], [Name 3]

---

# Problem Statement
- Traditional plagiarism checks rely on exact keyword matching
- Paraphrasing hides copied meaning
- Academia, legal, and content platforms need semantic detection

**Goal:** Detect meaning-level similarity across documents

---

# Solution Overview
- Convert sentences to semantic embeddings
- Compare document meaning using cosine similarity
- Output overall similarity + top matching sentences

**Key Output:** Interpretable similarity report

---

# Why Approach 3
- No model training required
- Strong accuracy for paraphrasing
- Fast enough for real-time demo
- Uses pre-trained SentenceTransformers

---

# System Architecture
```
Upload Documents
   ↓
Text Extraction (PDF/TXT/DOCX)
   ↓
Sentence Split
   ↓
Sentence Embeddings (SBERT)
   ↓
Cosine Similarity
   ↓
Similarity Report + Top Matches
```

---

# Core Pipeline (Step-by-Step)
1. Upload files
2. Extract raw text
3. Clean & normalize
4. Split into sentences
5. Generate embeddings
6. Compute cosine similarity
7. Report results

---

# Core Concepts
**Embeddings:** Meaning converted to numbers
**Vectorization:** Sentences → vectors
**Cosine Similarity:** Measures closeness (0–1)

**Why it works:** Paraphrases land near each other in embedding space

---

# Model & Libraries
**Model:** paraphrase-MiniLM-L3-v2 (SentenceTransformers, fast default)

**Libraries:**
- sentence-transformers
- scikit-learn
- numpy
- PyPDF2
- python-docx
- nltk
- streamlit

---

# Library Snippets (What Each Does)

**sentence-transformers** — Turn sentences into semantic vectors
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
embeddings = model.encode(["Hello world"], normalize_embeddings=True)
```

**scikit-learn** — Cosine similarity between embeddings
```python
from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0]
```

**numpy** — Vector math and averaging
```python
import numpy as np
doc_vec = np.mean(sentence_vectors, axis=0)
```

**PyPDF2** — Extract text from PDFs
```python
from PyPDF2 import PdfReader
text = "\n".join(p.extract_text() or "" for p in PdfReader("a.pdf").pages)
```

**python-docx** — Extract text from Word files
```python
from docx import Document
text = "\n".join(p.text for p in Document("a.docx").paragraphs)
```

**nltk** — Split text into sentences
```python
from nltk import sent_tokenize
sentences = sent_tokenize(text)
```

**streamlit** — Build the web UI
```python
import streamlit as st
files = st.file_uploader("Upload", accept_multiple_files=True)
```

---

# Tools (Logos)
![Python](https://www.python.org/static/community_logos/python-logo.png)
![SentenceTransformers](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/sbert_logo.png)
![scikit-learn](https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg)
![NumPy](https://numpy.org/images/logo.svg)
![Streamlit](https://streamlit.io/images/brand/streamlit-mark-color.png)

---

# Demo Flow
1. Upload two documents (one paraphrased)
2. Click **Compare**
3. View similarity score
4. Inspect top matching sentences
5. Download JSON report

---

# Example (Paraphrase)
**Doc A:** “The cat sat on the mat.”
**Doc B:** “A feline rested on the rug.”

**Keyword match:** Low
**Embedding similarity:** High

---

# Evaluation & Scoring
- **Similarity %** (0–100)
- **Sentence match threshold** controls sensitivity

**Interpretation:**
- 80–100%: High overlap
- 60–80%: Moderate
- 0–60%: Low

---

# Use Cases
- Academic plagiarism checks
- Legal document comparison
- Content duplication monitoring
- Internal document reuse auditing

---

# Limitations
- OCR errors in scanned PDFs
- Very short documents can be noisy
- Domain-specific jargon may reduce similarity

---

# Future Improvements
- Multilingual support
- Section/paragraph alignment
- Heatmap visualization
- Direct PDF highlighting

---

# Ethics & Responsible Use
- Assists reviewers, does not replace humans
- Similarity is a signal, not a verdict

---

# Team & Credits
- NLP/Model Integration: [Name]
- Backend/Engine: [Name]
- UI/Streamlit: [Name]
- Testing/Docs: [Name]

---

# Q&A
Thank you!
