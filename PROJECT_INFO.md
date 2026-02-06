# Plagiarism Detection (Approach 3) — Project Info

## Slide-Ready Outline (10+ Slides)
1. Title & Team
2. Problem Statement
3. Solution Overview
4. Why Semantic Similarity (Approach 3)
5. System Architecture
6. Core Pipeline (Step-by-Step)
7. Model & Libraries
8. Demo Flow
9. Evaluation & Results
10. Limitations & Future Work
11. Ethics & Responsible Use
12. Q&A

## Project Summary
This project detects semantic similarity and potential plagiarism between two or more documents (PDF/TXT/DOCX). It uses a pre-trained SentenceTransformers model to generate embeddings, then calculates cosine similarity to measure how close the documents are in meaning, even if the text is paraphrased.

### Title Slide (Suggested)
**Title:** Semantic Plagiarism Detection using Sentence Embeddings
**Subtitle:** Detecting paraphrased similarity across PDF/TXT/DOCX
**Hackathon:** [Your Hackathon Name]
**Team:** [Your Team Name]

## Objectives
- Perform semantic comparison of text documents
- Generate similarity scores (0–100%)
- Detect paraphrased or reworded plagiarism
- Produce a similarity report with top matching sentences

## Problem Statement (Slide)
Academic, legal, and content platforms need reliable tools to detect plagiarism beyond exact keyword matching. Traditional methods fail on paraphrasing.

## Solution Overview (Slide)
We convert sentences into semantic embeddings and compare document meaning using cosine similarity. The system outputs similarity scores plus top matching sentences for interpretability.

## Input
- Two or more documents: PDF / TXT / DOCX

## Output
- Similarity percentage between each document pair
- Highlighted or summarized similarity report (top matching sentences)
- Explanation of embedding model used

## AI / ML Techniques Used
- Sentence embeddings (pre-trained transformer)
- Sentence and document vectorization
- Cosine similarity for scoring

## Why Approach 3 (Slide)
- High accuracy for paraphrasing
- Fast enough for hackathon demo
- No model training required
- Strong real-world performance

## Core Concepts (Beginner-Friendly)

### 1. NLP Embeddings (The Magic Trick)
Think of embeddings like converting words into coordinates on a map:
- The word "dog" becomes numbers like `[0.2, -0.5, 0.8, ...]`
- The word "puppy" becomes similar numbers because they mean similar things
- Words with different meanings get different coordinates

**Why?** Computers can only work with numbers, not words. Embeddings convert meaning into numbers.

### 2. Sentence Vectorization
- Instead of converting single words, convert entire sentences into one set of numbers
- This captures the meaning of the whole sentence
- Similar sentences end up with similar vectors even if the wording is different

### 3. Cosine Similarity
A formula that compares two sets of numbers (vectors) and says: "How close are these on a scale of 0 to 1?"
- **1.0** = Identical meaning
- **0.5** = Somewhat similar
- **0.0** = Completely different

## Approach 3 (Chosen)
We use a pre-trained model to avoid training from scratch:
1. Extract text from documents
2. Split text into sentences
3. Convert sentences to embeddings
4. Compute cosine similarity
5. Generate report with similarity scores and top matching sentences

## Model Used
- **SentenceTransformers**: `all-MiniLM-L6-v2`
  - Fast, light-weight, strong semantic performance

## Tools & Logos (Slide)
Below are logos you can embed in your slides (URLs are safe to use in PPT converters):

![Python Logo](https://www.python.org/static/community_logos/python-logo.png)
![SentenceTransformers Logo](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/sbert_logo.png)
![scikit-learn Logo](https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg)
![NumPy Logo](https://numpy.org/images/logo.svg)
![Streamlit Logo](https://streamlit.io/images/brand/streamlit-mark-color.png)

## Libraries (Core)
- sentence-transformers
- scikit-learn
- numpy
- PyPDF2
- python-docx
- nltk
- streamlit

## Library Usage (With Tiny Code Snippets)

### sentence-transformers
Used to convert sentences into semantic embeddings.
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
```

### scikit-learn
Used for cosine similarity between vectors.
```python
from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0]
```

### numpy
Used for vector math and averaging embeddings.
```python
import numpy as np
doc_vec = np.mean(sentence_vecs, axis=0)
```

### PyPDF2
Used to extract text from PDF files.
```python
from PyPDF2 import PdfReader
text = PdfReader("sample.pdf").pages[0].extract_text()
```

### python-docx
Used to extract text from DOCX files.
```python
from docx import Document
text = "\n".join(p.text for p in Document("sample.docx").paragraphs)
```

### nltk
Used for sentence splitting (tokenization).
```python
from nltk import sent_tokenize
sentences = sent_tokenize(text)
```

### streamlit
Used to build the quick web UI for uploads and results.
```python
import streamlit as st
st.file_uploader("Upload documents", accept_multiple_files=True)
```

## Project Structure
```
Acumen/
├── similarity_engine.py     # Core semantic similarity logic
├── main.py                  # CLI runner
├── streamlit_app.py         # Web UI (Streamlit)
├── requirements.txt         # Dependencies
├── PROJECT_INFO.md          # This file
└── PLAGIARISM_DETECTION_GUIDE.md
```

## How Similarity Is Calculated
- Each sentence is encoded into a vector (embedding)
- Sentence embeddings are averaged to form a document embedding
- Cosine similarity is applied to get a document-level score
- Sentence-level matches are reported for interpretability

## System Architecture (Slide)
```
Client (Streamlit UI)
  → File Upload
  → Text Extraction (PDF/TXT/DOCX)
  → Sentence Split
  → Embedding Model (SBERT)
  → Similarity Scoring (Cosine)
  → Report (Scores + Matches)
```

## Working (Step-by-Step Flow)
1. **User uploads documents** (PDF/TXT/DOCX).
2. **Text extraction** reads and merges all pages/paragraphs into plain text.
3. **Cleaning** removes extra spaces and normalizes text.
4. **Sentence splitting** converts the document into a list of sentences.
5. **Embedding generation** converts each sentence into a numerical vector using the pre-trained model.
6. **Document embedding** is computed by averaging all sentence vectors.
7. **Cosine similarity** compares document embeddings to produce a score (0–100%).
8. **Sentence-level matching** compares each sentence from Document A to Document B and keeps top matches above a threshold.
9. **Report output** shows overall similarity plus top matching sentence pairs.

### Working (Beginner-Friendly Explanation)
- Think of each sentence as being turned into a “meaning fingerprint.”
- Sentences with similar meaning end up with similar fingerprints, even if words are changed.
- We compare those fingerprints using a math formula (cosine similarity).
- Finally, we summarize: **overall similarity** + **where the strongest matches are**.

### Flow Diagram (Text)
```
Upload Documents
  ↓
Extract Text (PDF/TXT/DOCX)
  ↓
Clean + Split into Sentences
  ↓
Sentence Embeddings (AI Model)
  ↓
Cosine Similarity (Score)
  ↓
Top Matching Sentences + Report
```

## What Makes It Detect Paraphrasing
- The embedding model captures meaning, not just exact words
- Paraphrased sentences land close to each other in embedding space
- Cosine similarity measures that closeness even if wording is different

## Demo Flow (Slide)
1. Upload two documents (one paraphrased).
2. Click **Compare Documents**.
3. View similarity score.
4. Expand top matching sentences.
5. Download JSON report.

## Example (Simple)
Document A: "The cat sat on the mat."
Document B: "A feline rested on the rug."

- Keyword match: low
- Embedding similarity: high (same meaning)

## Report Contents
- Pairwise similarity scores for all document combinations
- Top matching sentence pairs (with scores)
- Model name used and similarity method
- Similarity score report
- Explanation of embedding model used
- UI for easy testing (Streamlit)

## Notes
- No model training required
- Works best with clean text extraction
- First run downloads the model

## Evaluation & Results (Slide)
- **Metric:** Cosine similarity (0–100%)
- **Interpretation:**
  - 80–100%: High similarity / likely paraphrase
  - 60–80%: Moderate overlap
  - 0–60%: Low similarity

## Accuracy Considerations
- Embeddings are strong for paraphrases
- Results depend on text extraction quality
- More sentences = more stable scores

## Limitations (Slide)
- OCR errors in scanned PDFs reduce accuracy
- Very short documents can be noisy
- Domain-specific jargon can reduce similarity

## Future Improvements (Slide)
- Multilingual support
- Section-level alignment
- Visual heatmap of matches
- PDF highlighting in UI

## Ethics & Responsible Use (Slide)
- Assists reviewers, does not replace human judgment
- Similarity score is a signal, not a final verdict

## Team Roles (Optional Slide)
- NLP/Model Integration: [Name]
- Backend/Similarity Engine: [Name]
- UI/Streamlit: [Name]
- Testing & Documentation: [Name]

---

## Hackathon Pitch (Quick Summary)
**Problem:** Detect plagiarism beyond keyword matching.
**Solution:** Use sentence embeddings + cosine similarity to compare document meaning.
**Impact:** Helps academia, legal, and content platforms identify paraphrased reuse.

## Key Features
- Supports PDF, TXT, DOCX
- Semantic (meaning-based) comparison
- Sentence-level matching highlights
- Pairwise similarity scores for multiple documents
- Exportable JSON report

## What Makes It Innovative
- Goes beyond exact text matching
- Catches paraphrased content using semantic embeddings
- Interpretable output with top matching sentence pairs

## Demo Plan (2–3 Minutes)
1. Upload two documents (one paraphrased version).
2. Run comparison.
3. Show similarity score.
4. Expand top matches to demonstrate paraphrase detection.
5. Download report.

## Evaluation Metrics (For Judges)
- **Similarity %**: Overall semantic overlap
- **Sentence Match Score**: Localized similarity
- **Threshold**: Controls sensitivity to paraphrasing

## Use Cases
- Academic plagiarism screening
- Legal document similarity
- News/content duplication checks
- Internal document reuse auditing

## Limitations (Transparency)
- PDF extraction quality affects results
- Short documents may produce unstable scores
- Highly technical jargon can reduce embedding quality

## Future Improvements
- Add language detection + multilingual models
- Section/paragraph-level alignment
- Plagiarism heatmap visualization
- PDF highlighting directly in the UI
- Caching for faster repeated comparisons

## Ethical Considerations
- This tool assists review; it does not replace human judgment
- Similarity score is an indicator, not a final verdict

## How to Run (Quick)
1. Install dependencies: `pip install -r requirements.txt`
2. Start app: `streamlit run streamlit_app.py`
