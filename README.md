# AstroTrust-AI

AstroTrust-AI is a research-oriented prototype for uncertainty-aware classification, novelty detection, and follow-up prioritization in astronomical alert streams.

The current version uses ELAsTiCC2 light-curve data and implements a complete MVP pipeline:

1. data loading and preprocessing;
2. light-curve feature extraction;
3. supervised classification with Random Forest;
4. early classification using partial light curves;
5. uncertainty estimation;
6. novelty detection;
7. follow-up prioritization;
8. Streamlit dashboard for candidate inspection.

## Project Structure

```text
astrotrust-ai/
├── data/
│   ├── raw/
│   └── processed/
├── experiments/
├── results/
├── scripts/
├── src/
│   ├── data/
│   ├── features/
│   ├── interface/
│   └── models/
├── requirements.txt
└── README.md