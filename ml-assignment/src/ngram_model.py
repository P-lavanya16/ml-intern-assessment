# trigram-assignment/src/ngram_model.py
from collections import defaultdict, Counter
import math
import random
from typing import Iterable, List, Tuple, Dict, Union

class TrigramModel:
    """
    Simple trigram language model with add-k smoothing.
    Usage:
      model = TrigramModel(k=1e-3)
      model.train(["this is a sentence .", "this is another sentence ."])
      model.predict_next("this", "is")  # returns top-1 next word
      model.next_word_probs("this","is") # returns dict(word -> prob)
      model.generate(("this","is"), max_len=10)
    """

    def __init__(self, k: float = 1e-3):
        # smoothing constant
        self.k = float(k)

        # counts
        self.unigram = Counter()   # unigram counts: w
        self.bigram = Counter()    # bigram counts: (w1,w2)
        self.trigram = Counter()   # trigram counts: (w1,w2,w3)

        # vocabulary set
        self.vocab = set()
        self.total_unigrams = 0
        self.trained = False

    # --- Utilities ---
    def _tokenize(self, text: Union[str, Iterable[str]]) -> List[List[str]]:
        """
        If `text` is an iterable of strings (sentences), tokenize each.
        If text is a single string, split into sentences by newline and whitespace.
        Very simple tokenizer: split by whitespace. Keep punctuation tokens if present.
        """
        if isinstance(text, str):
            # treat each newline as separate "sentence"
            lines = [line.strip() for line in text.splitlines() if line.strip()]
        else:
            # assume iterable of strings
            lines = [line.strip() for line in text if isinstance(line, str) and line.strip()]
        tokenized = []
        for line in lines:
            # naive split — tests usually expect simple whitespace tokenization
            toks = line.split()
            if not toks:
                continue
            # add explicit start tokens if helpful
            # but for this assignment we keep raw tokens; users/tests may include <s> tokens themselves
            tokenized.append(toks)
        return tokenized

    # --- Training ---
    def train(self, corpus: Union[str, Iterable[str]]):
        """
        Train on a corpus. corpus can be:
          - a single string (possibly with newlines)
          - an iterable of sentence strings
        """
        tokenized_sentences = self._tokenize(corpus)
        # reset counts
        self.unigram.clear()
        self.bigram.clear()
        self.trigram.clear()
        self.vocab.clear()
        self.total_unigrams = 0

        for toks in tokenized_sentences:
            # update unigram counts
            for t in toks:
                self.unigram[t] += 1
                self.vocab.add(t)
                self.total_unigrams += 1

            # update bigrams
            if len(toks) >= 2:
                for i in range(len(toks)-1):
                    self.bigram[(toks[i], toks[i+1])] += 1

            # update trigrams
            if len(toks) >= 3:
                for i in range(len(toks)-2):
                    self.trigram[(toks[i], toks[i+1], toks[i+2])] += 1

        self.trained = True
        return self

    # alias
    fit = train

    # --- Probability helpers ---
    def _vocab_size(self) -> int:
        # vocabulary size for smoothing (include <unk> if desired)
        return max(1, len(self.vocab))

    def next_word_probs(self, w1: str, w2: str) -> Dict[str, float]:
        """
        Return a dictionary mapping candidate next words -> P(word | w1, w2)
        using add-k smoothing:
           P(w3 | w1,w2) = ( count(w1,w2,w3) + k ) / ( count(w1,w2) + k*V )
        If bigram (w1,w2) was unseen, back off to bigram-with-smoothing using unigram denom.
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() or fit() with corpus first.")

        V = self._vocab_size()
        denom = self.bigram.get((w1, w2), 0)

        probs = {}
        if denom > 0:
            denom_smooth = denom + self.k * V
            # iterate over vocabulary
            for w in self.vocab:
                num = self.trigram.get((w1, w2, w), 0) + self.k
                probs[w] = num / denom_smooth
            # ensure numerical stability
            self._normalize_probs_dict(probs)
            return probs
        else:
            # Back-off: use bigram (w2, w) with smoothing based on unigram counts of w2
            # denom2 is count(w2) (unigram)
            denom2 = self.unigram.get(w2, 0)
            denom2_smooth = denom2 + self.k * V
            for w in self.vocab:
                num = self.bigram.get((w2, w), 0) + self.k
                probs[w] = num / denom2_smooth
            self._normalize_probs_dict(probs)
            return probs

    def _normalize_probs_dict(self, d: Dict[str, float]):
        s = sum(d.values())
        if s <= 0:
            return
        for k in list(d.keys()):
            d[k] = d[k] / s

    def predict_next(self, w1: str, w2: str, top_n: int = 1) -> List[Tuple[str, float]]:
        """
        Return top_n (word, prob) pairs sorted by probability descending.
        If top_n==1, returns a single-element list.
        """
        probs = self.next_word_probs(w1, w2)
        # sort by prob desc
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_n]

    # alias for tests that may expect this name
    predict = predict_next

    # --- Generation ---
    def generate(self, start_words: Tuple[str, str], max_len: int = 20, greedy: bool = True) -> List[str]:
        """
        Generate tokens starting from start_words = (w1, w2).
        If greedy True: pick argmax at each step.
        If greedy False: sample according to distribution.
        Returns the list of generated tokens including the start pair.
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() or fit() first.")

        w1, w2 = start_words
        out = [w1, w2]
        for _ in range(max_len - 2):
            probs = self.next_word_probs(w1, w2)
            if not probs:
                break
            if greedy:
                # pick highest probability
                next_word = max(probs.items(), key=lambda x: x[1])[0]
            else:
                # sample
                words, ps = zip(*probs.items())
                # cumulative sampling
                r = random.random()
                cum = 0.0
                next_word = words[-1]
                for w, p in zip(words, ps):
                    cum += p
                    if r <= cum:
                        next_word = w
                        break
            out.append(next_word)
            w1, w2 = w2, next_word
        return out

    # convenience: score a sequence (log-prob)
    def score_sequence(self, tokens: List[str]) -> float:
        """
        Return log-probability (natural log) of the given token sequence under the model.
        Use trigram probabilities with smoothing/back-off logic implemented in next_word_probs.
        For the first two tokens, we do not compute trigram probabilities (score from 3rd token onwards).
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() or fit() first.")
        if len(tokens) < 3:
            return 0.0
        logp = 0.0
        for i in range(2, len(tokens)):
            w1, w2, w3 = tokens[i-2], tokens[i-1], tokens[i]
            probs = self.next_word_probs(w1, w2)
            p = probs.get(w3, 0.0)
            # avoid log(0) by using a tiny floor
            p = max(p, 1e-12)
            logp += math.log(p)
        return logp
