# Plagiarism Detection System - Complete Hackathon Guide

## Table of Contents
1. [What You're Building](#what-youre-building)
2. [Core Concepts](#core-concepts)
3. [Possible Approaches](#possible-approaches)
4. [Recommended Approach (Approach 3)](#recommended-approach)
5. [Approach 3: Detailed 5-Step Process](#approach-3-detailed-5-step-process)
6. [Complete Skills & Knowledge Required](#complete-skills--knowledge-required)
7. [Tech Stack](#tech-stack)
8. [Timeline Breakdown](#timeline-breakdown)
9. [Getting Started](#getting-started)

---

## What You're Building

Imagine you have two essays. You want to know: *"How similar are these two texts?"* Not just word-for-word matching, but even if someone rephrased the ideas. Your system needs to be smart enough to catch paraphrased plagiarism.

### Problem Statement
- Develop an intelligent system that compares two or more documents
- Measure semantic similarity beyond exact keyword matching
- Identify potential plagiarism or duplicated content (including paraphrased)
- Support multiple file formats (PDF, TXT, DOCX)

### Expected Output
- Similarity percentage between documents
- Highlighted or summarized similarity report
- Explanation of embedding model used

---

## Core Concepts

### 1. NLP Embeddings (The Magic Trick)
Think of embeddings like converting words into coordinates on a map:
- The word "dog" becomes numbers like `[0.2, -0.5, 0.8, ...]`
- The word "puppy" becomes similar numbers because they mean similar things
- Words with different meanings get different coordinates

**Why?** Computers can only work with numbers, not words. Embeddings convert meaning into numbers.

### 2. Sentence Vectorization
- Instead of converting single words, convert entire sentences into one set of numbers
- This captures the meaning of the whole sentence

### 3. Cosine Similarity
A formula that compares two sets of numbers (vectors) and says: "How close are these on a scale of 0 to 1?"
- **1.0** = Identical meaning
- **0.5** = Somewhat similar
- **0.0** = Completely different

---

## Possible Approaches

### Approach 1: Simple Keyword-Based (Easiest, Fastest)
```
How it works:
Document 1 → Extract keywords
Document 2 → Extract keywords
Compare keywords → Get similarity score

Time: 2-3 hours
Accuracy: 40-50% (catches obvious plagiarism)
Pros: Super fast to code, works offline
Cons: Misses paraphrased content, many false positives
```

### Approach 2: TF-IDF Based (Beginner-Friendly)
```
How it works:
- Convert documents to word frequency vectors
- Calculate cosine similarity between vectors
- Report similarity percentage

Tools: sklearn library (very easy to use)
Time: 4-5 hours
Accuracy: 60-70%
Pros: Detects some paraphrasing, simple to implement
Cons: Still misses semantic meaning
```

### Approach 3: Pre-trained Sentence Embeddings (Smart & Practical) ⭐ RECOMMENDED
```
How it works:
- Use SentenceTransformers (pre-trained AI model)
- Split documents into sentences
- Convert each sentence to embeddings
- Compare embeddings using cosine similarity
- Generate detailed report

Tools: SentenceTransformers, PyPDF2 (for PDFs)
Time: 6-8 hours
Accuracy: 80-90%
Pros: Catches paraphrasing, understands meaning, fast
Cons: Needs internet first time (to download model)
```

### Approach 4: Advanced Multi-Stage (Most Powerful, Complex)
```
How it works:
- Segment documents (by paragraphs/sections)
- Detect plagiarism at multiple levels
- Generate detailed similarity maps
- Show which parts match
- Use advanced metrics (Jaccard, alignment scoring)

Time: 12-16 hours
Accuracy: 90%+
Pros: Most accurate, detailed reports
Cons: Overkill for 2-day deadline
```

---

## Recommended Approach

### Why Approach 3 for Your 2-Day Deadline?

| Factor | Why This Works |
|--------|---------------|
| **Time** | 6-8 hours coding = safe margin for testing |
| **Accuracy** | 80-90% catches paraphrasing (main goal) |
| **Ease** | 95% of work is using existing libraries |
| **Features** | Has all the "wow factor" for demo |
| **Scalability** | Can improve even after submission |

### High-Level Architecture

```
INPUT (PDF/TXT/DOCX)
        ↓
   [Extract Text]
        ↓
   [Split into Sentences]
        ↓
   [Convert to Embeddings] ← Uses SentenceTransformers
        ↓
   [Calculate Cosine Similarity]
        ↓
   [Generate Report] → Output similarity % + highlights
```

---

## Approach 3: Detailed 5-Step Process

### Step 1: Load Documents
- Read PDF/TXT/DOCX files
- Extract all text content
- Clean up formatting

### Step 2: Break Into Sentences
- Split each document into individual sentences
- Example: "Hello world. This is a test." → 2 sentences

### Step 3: Convert to Embeddings
- Use SentenceTransformers model (already trained by researchers)
- Each sentence becomes a vector (list of ~384 numbers)
- Similar sentences get similar number patterns

### Step 4: Calculate Similarity
- Compare sentence vectors using cosine similarity formula
- Get a similarity score (0-100%)
- Find which sentences/paragraphs match

### Step 5: Generate Report
- Display overall similarity percentage
- Show which parts of documents are similar
- Explain which model was used

### In Code Terms (What We'll Write)

```
1. Function to read documents
   
2. Function to split text into sentences
   
3. Function to convert sentences to embeddings
   (using pre-trained model - we don't train it)
   
4. Function to calculate similarity between embeddings
   
5. Function to generate a nice report with results
   
6. Web interface to upload documents and see results
```

### The Key Insight

We're **NOT training an AI model**. We're using a **pre-trained, ready-to-use model** from SentenceTransformers that already understands English. We just leverage it.

It's like using Google Translate - you don't build it, you just use their API.

---

## Complete Skills & Knowledge Required

### MUST KNOW (Core to Project)

#### 1. Python Basics (Essential)
```python
# Variables & data types
name = "John"
age = 25
scores = [90, 85, 88]  # lists

# Functions
def compare_documents(doc1, doc2):
    return similarity_score

# Loops
for sentence in sentences:
    convert_to_embedding(sentence)

# If-else statements
if similarity > 0.7:
    print("Plagiarism detected!")
```
**Time to learn:** If you know basic programming, you're fine. If not: 2-3 hours.

#### 2. Libraries You'll Use (NOT Code from Scratch)

##### SentenceTransformers
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # That's it!
embeddings = model.encode(sentences)  # Magic happens here
```
**What you need to know:** How to use it (not how it works internally). Like using Google Maps API.

##### Scikit-learn (For Similarity)
```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(embedding1, embedding2)
```
**What you need to know:** This calculates how similar two vectors are (0-1 scale).

##### PyPDF2 / python-docx (For Reading Files)
```python
from PyPDF2 import PdfReader

pdf = PdfReader("document.pdf")
text = pdf.pages[0].extract_text()
```
**What you need to know:** How to extract text from files.

##### NumPy (Math Library)
```python
import numpy as np

array = np.array([1, 2, 3])  # Like lists but for math
```
**What you need to know:** Basics of arrays. Pre-trained models handle the complex math.

### NICE TO HAVE (For Web Interface)

#### Option A: Streamlit (EASIEST - Recommended)
```python
import streamlit as st

st.title("Plagiarism Detector")
uploaded_files = st.file_uploader("Upload documents")
if st.button("Check Similarity"):
    result = compare_documents(uploaded_files)
    st.write(result)
```
**What you need to know:** 
- Basic Python (that's it!)
- NO HTML/CSS/JavaScript needed
- Streamlit handles the web interface for you

**Learning time:** 1 hour

#### Option B: Flask (INTERMEDIATE)
```python
from flask import Flask, render_template, request

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files['documents']
    result = compare_documents(files)
    return render_template('result.html', result=result)
```

**What you need to know:**
- Basic Python
- **HTML basics** (form structure)
- **CSS basics** (styling)
- **JavaScript basics** (handling uploads, showing results)

**Learning time:** 3-4 hours

#### Option C: Full Stack (NOT RECOMMENDED for 2-day deadline)
```
React.js / Vue.js (Complex JavaScript frameworks)
Backend API architecture
Database design
etc.
```
**Learning time:** 20+ hours - **SKIP THIS**

### Knowledge Map (What to Learn)

#### Tier 1: MUST HAVE (Non-Negotiable)
```
✅ Python fundamentals
   ├─ Variables, functions, loops, conditionals
   ├─ Lists, dictionaries
   └─ Error handling (try-except)

✅ Using libraries (NOT building them)
   ├─ How to import
   ├─ How to call functions
   └─ Reading documentation

✅ File handling
   └─ Reading text, PDF, Word files

✅ Basic math concepts
   └─ Understanding vectors & similarity (we explain the formula)
```

#### Tier 2: WEB INTERFACE (Pick One)

**If you choose Streamlit (EASIEST):**
```
✅ Streamlit basics only
   ├─ st.file_uploader
   ├─ st.button
   ├─ st.write
   └─ st.progress (for status)
```
**Learning time:** 30 mins - 1 hour

**If you choose Flask:**
```
✅ HTML basics
   ├─ Form structure (<form>, <input>)
   ├─ File upload forms
   └─ Display results

✅ CSS basics
   ├─ Basic styling
   ├─ Responsive design
   └─ Colors, fonts, spacing

✅ JavaScript basics
   ├─ Handling form submission
   ├─ Showing loading indicator
   ├─ Displaying results
   └─ Basic AJAX (sending data without refresh)

✅ Flask basics
   ├─ Routing (@app.route)
   ├─ Handling POST requests
   └─ Returning HTML
```
**Learning time:** 3-4 hours

#### Tier 3: OPTIONAL (Nice extras, not needed)
```
❌ Advanced JavaScript frameworks (React, Vue)
❌ Database design (SQLite, PostgreSQL)
❌ API architecture
❌ Authentication & security (unless required)
❌ Cloud deployment
```

### What You DON'T Need to Know

| Thing | Why You Don't Need It |
|-------|----------------------|
| How transformer models work | Pre-trained model handles it |
| Linear algebra details | NumPy handles the math |
| How embedding algorithms work | SentenceTransformers abstracts it |
| Advanced JavaScript | Use Streamlit instead |
| Database | Store results in files/JSON |
| DevOps / Docker | Run locally for demo |

---

## Tech Stack

### Core Tools (MUST HAVE)
```
Python 3.9+
├── sentence-transformers       (AI embeddings)
├── scikit-learn                (similarity calculations)
├── PyPDF2 / python-docx        (read PDF/DOCX)
├── nltk                        (text processing)
└── numpy                       (mathematical operations)
```

### Web Interface (Pick One)
```
Option 1: Streamlit (Recommended for speed)
  └─ Just Python, no HTML/CSS/JS

Option 2: Flask (More control)
  ├─ Flask (Python backend)
  ├─ HTML (frontend)
  ├─ CSS (styling)
  └─ JavaScript (interactivity)
```

### Installation Command
```bash
# Core dependencies
pip install sentence-transformers scikit-learn PyPDF2 python-docx nltk numpy

# For Streamlit
pip install streamlit

# For Flask (if you choose it)
pip install flask
```

---

## Timeline Breakdown

### Day 1 (Recommended Schedule)
```
├─ 1 hour: Setup & environment configuration
├─ 2 hours: Core engine (document loading + embedding)
├─ 2 hours: Similarity calculation & report generation
└─ 1 hour: Testing with sample documents
```

### Day 2 (Recommended Schedule)
```
├─ 1 hour: Web interface setup (Streamlit or Flask)
├─ 1.5 hours: Integration & debugging
├─ 0.5 hour: Testing with real documents
├─ 0.5 hour: Optimization & polish
└─ 1 hour: Documentation & presentation prep
```

**Total Coding Time:** ~8-10 hours (with buffer for debugging)

---

## Getting Started

### Prerequisites Check
Have you:
- ✓ Used Python before?
- ✓ Installed Python packages with pip?
- ✓ Ever written a function?

If yes to all 3 → You're ready to code right now!
If no → Get 30-min Python crash course first

### My Recommendation for 2-Day Deadline

**GO WITH STREAMLIT + Python only**
```
Reasons:
├─ Reason 1: No HTML/CSS/JavaScript needed
├─ Reason 2: Built for data science projects
├─ Reason 3: Perfect for demos
├─ Reason 4: Learn it in 1 hour
└─ Reason 5: Focus on core logic, not UI
```

**Code looks like:**
```python
import streamlit as st
from sentence_transformers import SentenceTransformer

st.title("Plagiarism Detector")

files = st.file_uploader("Upload 2+ documents", accept_multiple_files=True)
if st.button("Check Similarity"):
    similarity = compare_documents(files)
    st.success(f"Similarity: {similarity}%")
```

That's it! The Streamlit framework handles everything else.

### Learning Path (In Order)

**Day 1:**
1. Python basics review (30 mins)
2. Learn SentenceTransformers + scikit-learn (1 hour)
3. Build core similarity engine (3 hours)
4. Test with sample documents (1 hour)

**Day 2:**
1. Learn Streamlit (30 mins)
2. Build web interface (1 hour)
3. Integration + testing (2 hours)
4. Documentation (1 hour)

### Resources You'll Need

- **Python 3.9+** (check: `python --version`)
- **PyCharm / VS Code** (code editor)
- **Sample documents** (test PDFs/TXT files)
- **Internet connection** (first run downloads the pre-trained model)

---

## Project Structure (What We'll Build)

```
plagiarism-detector/
├── main.py                      # Entry point
├── similarity_engine.py          # Core logic
│   ├── load_documents()
│   ├── preprocess_text()
│   ├── get_embeddings()
│   ├── calculate_similarity()
│   └── generate_report()
├── streamlit_app.py             # Web interface
├── requirements.txt             # Dependencies
├── sample_documents/            # Test files
│   ├── doc1.txt
│   ├── doc2.txt
│   └── doc3.pdf
└── README.md                    # Documentation
```

---

## Key Takeaways

1. **You don't train the AI** - You use a pre-trained model (like a spell-checker app)
2. **It's mostly gluing libraries together** - 80% is library usage, 20% is your code
3. **Accuracy comes "free"** - The SentenceTransformers model is already trained on billions of examples
4. **Plagiarism ≠ Exact Match** - Your system catches *meaning* similarity, not just word matching
5. **Use Streamlit** - Saves massive amounts of time on the web interface
6. **Test with multiple documents** - PDF, TXT, DOCX formats
7. **Document your findings** - Explain which model you used and why

---

## Deliverables Checklist

- [ ] Document similarity engine (Python)
- [ ] Similarity score calculation (0-100%)
- [ ] Support for multiple file formats (PDF, TXT, DOCX)
- [ ] Web interface (Streamlit)
- [ ] Sample test documents
- [ ] Detailed README explaining:
  - How to run the project
  - What model was used (all-MiniLM-L6-v2)
  - Why this approach was chosen
  - Results and accuracy metrics
- [ ] Demo with real documents
- [ ] Presentation slides

---

## Next Steps

Once you're ready to code, we'll:
1. Set up the project structure
2. Install required libraries
3. Build the similarity engine
4. Create the web interface
5. Test with sample documents
6. Generate documentation

**Say "Let's code!" when you're ready to start implementing!**

---

*Last Updated: February 5, 2026*
*Hackathon Deadline: 48 hours*
