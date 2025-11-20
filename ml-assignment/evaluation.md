# Evaluation & Design Notes

**Model implemented:** `TrigramModel` — a simple trigram language model with add-k smoothing.

**Design choices**
- **Tokenization:** Minimal whitespace-based tokenization to match typical unit tests for this exercise. This keeps behaviour deterministic and simple.
- **Counts:** Store unigram, bigram and trigram counts in `collections.Counter`.
- **Smoothing:** Add-k (Laplace) smoothing with a small default `k=1e-3`. This prevents zero probabilities while not washing out observed counts.
- **Back-off:** If a trigram context `(w1,w2)` is unseen, the model backs off to the bigram `(w2,w)` distribution (also with add-k smoothing). This is a lightweight back-off that works well for small data.
- **API:** Implemented `train`/`fit`, `next_word_probs`, `predict_next`/`predict`, `generate`, and `score_sequence`. Multiple method names added to match possible test expectations.

**How to run**
1. Install requirements: `pip install -r requirements.txt`
2. From repository root run tests: `pytest trigram-assignment/tests/test_ngram.py -q`
3. Example usage:
   ```py
   from trigram_assignment.src.ngram_model import TrigramModel
   model = TrigramModel(k=1e-3)
   model.train(["this is a test .", "this is another test ."])
   print(model.predict_next("this","is"))
Notes / Improvements

Swap tokenizer for nltk or spacy if punctuation handling or sentence segmentation is required.

Replace add-k smoothing with Kneser-Ney or interpolated smoothing for improved performance on real corpora.

Add <s> and </s> tokens for explicit sentence boundaries if evaluation requires sentence-level probabilities.

---

### How to run tests locally
1. From repo root:
   - `pip install -r requirements.txt`
   - `pytest trigram-assignment/tests/test_ngram.py -q`
2. If pytest fails due to different expected method names, try importing and calling:
   - `TrigramModel.train(...)` or `TrigramModel.fit(...)`
   - `predict_next` or `predict`

---


