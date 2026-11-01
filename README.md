# HMS-PED: Hybrid Multi-Stage Phishing Email Detector

**Team:** Sameer Hembram (S24CSEU0453), Ayushmaan Kumar (S24CSEU2784) · Bennett University, Greater Noida

**Base paper:** E. Pham, T. Bui, H. Chen, "Demo: LLM-Based Browser Extension for Phishing Detection," IEEE SVCC 2025, [doi:10.1109/SVCC65277.2025.11133657](https://doi.org/10.1109/SVCC65277.2025.11133657)

**Reference framework:** K. Saurabh *et al.*, "HMS-IDS," *Arabian Journal for Science and Engineering*, 2025, [doi:10.1007/s13369-024-08935-5](https://doi.org/10.1007/s13369-024-08935-5)

A phishing-email detector that adapts the HMS-IDS hybrid intrusion-detection framework to email and
extends the base paper's LLM-only Gmail extension:

| Stage | What it does | Source |
|---|---|---|
| 1 | Stacking ensemble of Naive Bayes, Decision Tree, KNN, Logistic Regression on TF-IDF + Information-Gain features + 9 phishing indicators | adapted from HMS-IDS |
| 2 | CL-k-means anomaly layer: re-checks emails Stage 1 calls legitimate | adapted from HMS-IDS |
| 3 | Gemini LLM: re-checks *uncertain* emails and writes the explanation | prompt from the original extension |

> **Built on prior work.** The browser extension, popup and Gemini prompt come from the base paper's
> open-source code, [CMPE209FinalProject](https://github.com/Xainax/CMPE209FinalProject) by Eric Pham, Thinh Bui and
> Haoming Chen (San José State University, 2025). That work used the LLM alone (91% accuracy on 100 SpamAssassin emails).
> **Our additions:** the HMS-IDS-style ML stages, the train/validation/test datasets, the full
> evaluation (including the unseen-corpus test), the `/api/hybrid-scan` backend, and a fix for the
> popup's "Loading email..." bug.

## Results (held-out test split, 6,270 emails)

| Model | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| Naive Bayes | 91.87% | 0.9989 | 0.8473 | 0.9169 | 0.0010 |
| Decision Tree | 98.28% | 0.9823 | 0.9852 | 0.9838 | 0.0200 |
| KNN | 83.83% | 0.7663 | 0.9994 | 0.8675 | 0.3431 |
| Logistic Regression | 99.51% | 0.9949 | 0.9958 | 0.9953 | 0.0058 |
| **Stacking ensemble** | **99.54%** | 0.9958 | 0.9955 | **0.9956** | 0.0047 |
| CL-k-means alone | 98.25% | 0.9837 | 0.9831 | 0.9834 | 0.0183 |
| Hybrid (Stage 1 + 2) | 99.51% | 0.9952 | 0.9955 | 0.9953 | 0.0054 |

**Unseen-corpus test** (train on one corpus, test on the other): the stacking ensemble's F1 falls to
**0.3647** (CEAS → Ling) and **0.7920** (Ling → CEAS). Near-perfect in-distribution accuracy does not
transfer to unseen phishing campaigns, which is the main finding of the paper. Full numbers are in `results/`.

## Repository layout

```
data/raw/            Ling.csv.gz, CEAS_08.csv.gz  (original public corpora)
data/splits/         train.csv.gz, val.csv.gz, test.csv.gz  (70/15/15, stratified, seed 42)
src/                 all experiment code (run in the order below)
results/             JSON/CSV results + figures/ (PDF and PNG)
backend/             Flask API used by the extension (/api/hybrid-scan, /api/llm-query)
extension/           Chrome/Brave extension (load this folder unpacked)
frontend/            React source of the popup (built into extension/dist)
paper/               our draft research paper: LaTeX source (Overleaf) + PDF
papers/              base paper and related papers (PDF)
slides/              presentation slides
```

## 1. Set up (Windows, macOS or Linux, Python 3.11+)

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## 2. Reproduce the experiments (~10 minutes on a laptop)

```bash
cd src
python data_prep.py        # builds data/splits/ from data/raw/
python train_stage1.py     # Stage 1: tunes on validation, tests once on test
python stage2_clkmeans.py  # Stage 2: CL-k-means + hybrid
python cross_corpus.py     # unseen-corpus (zero-day-style) experiment
python make_figures.py     # results/figures/
python export_latex.py     # paper/ tables and numbers
```

## 3. LLM (Gemini) experiments, which need your own API key

1. Get a free key at <https://aistudio.google.com/>.
2. Copy `.env.example` to `.env` and paste the key into it. **Never commit `.env`.**
3. Run the following (answers are cached, so if you hit a rate limit just run the command again):

```bash
cd src
python gemini_eval.py --mode standalone --n 100   # Gemini vs ensemble on test emails
python gemini_eval.py --mode cross --n 100        # Gemini vs CEAS-trained ensemble on Ling
python gemini_eval.py --mode hybrid               # Gemini as Stage 3 on uncertain emails
python export_latex.py                            # adds the LLM table to the paper
```

## 4. Run the browser extension

1. Start the backend (keep this window open). The trained models must be in `models/`; if they are missing, run step 2 first.
   ```bash
   cd backend
   python app.py
   ```
2. In Chrome or Brave, open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked**, and choose the `extension` folder.
3. Open Gmail, **refresh the tab**, open an email, click the extension and then **Scan Email**.
   The result shows the verdict, the ML phishing probability and which stage decided.

To rebuild the popup after changing `frontend/phishing-extension/src/App.jsx`: `npm install && npm run build`
inside `frontend/phishing-extension` (Node 20+).

## Datasets

Ling and CEAS-2008 as curated by Champa, Rabbi and Zibran, *Phishing Email Curated Datasets*,
Zenodo, [doi:10.5281/zenodo.8339691](https://doi.org/10.5281/zenodo.8339691). Label 1 = phishing/malicious,
0 = legitimate. Note that these are older corpora and much of their malicious class is spam.

## Papers

- **Draft paper:** `paper/` (IEEE format, compiled on Overleaf).
- **Base paper:** `papers/` (Pham, Bui, Chen, IEEE SVCC 2025).
- **Also referenced:** Rahman, Rahman, Sun, IEEE NAPS 2025; HMS-IDS (reference framework).
