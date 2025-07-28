# SRVSKG: Sign-aware Recommendation based on Virtual Semantic Knowledge Graph

![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)


This repository contains the implementation for SRVSKG, a novel knowledge graph-based recommendation system that incorporates virtual semantic relations and sign-aware learning.

## Requirements

- Python 3.8
- Dependencies listed in `requirement.txt`

Install all required packages:
```
pip install -r requirement.txt
```

## Data Preparation

### Directory Structure
Create the following folder structure and place your dataset files accordingly:
```bash
./data/
├── amazon-book/
│   ├── train.txt        # Training data
│   ├── test.txt         # Test data
│   ├── ua_triplets.npy  # ua_triplets
│   └── kg_final.txt     # Knowledge graph triples
└── yelp/
    ├── train.txt
    ├── test.txt
    ├── ua_triplets.npy
    └── kg_final.txt
