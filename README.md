# TF-IDF-zh

## Introduction

`TF-IDF-zh` is a Python project focused on processing Chinese text, utilizing the TF-IDF (Term Frequency-Inverse Document Frequency) algorithm to assess the importance of words within a collection of documents. This project is suitable for applications in text mining, information retrieval, document classification, and more, aiming to provide an efficient and accurate tool for Chinese text analysis.

## Features

- **Text Processing**: Preprocess Chinese text, including tokenization, stop word removal, etc.
- **TF-IDF Calculation**: Compute the TF-IDF values to evaluate the significance of words in documents.
- **Versatile Applications**: Can be used to build search engines, recommendation systems, document classifiers, etc.

## Quick Start

### Prerequisites

- Python 3.x
- Required Python libraries: see `requirements.txt`

### Installation

Clone the repository first:

```bash
git clone https://github.com/ReeveWu/TF-IDF-zh.git
cd TF-IDF-zh
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage example

```python
from utilities.TfidfVectorizer import TfidfVectorizer

# Initialize TFIDF object
tfidf = TfidfVectorizer()

# Load document data
documents = ["Text of document 1...", "Text of document 2...", ...]

# Compute TF-IDF
tfidf_values = tfidf_values = tfidf.fit_transform(documents)

# Print results
print(tfidf_values)
```
