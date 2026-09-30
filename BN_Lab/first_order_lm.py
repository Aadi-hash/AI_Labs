"""First-order autoregressive language model (bigram / Bayesian chain X1 -> X2 -> ... -> XT).
Estimates P(X_t | X_{t-1}) from transition counts. Ordinary Python only."""
import random
from collections import defaultdict, Counter

START, END = "<START>", "<END>"

DATA = """the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug"""


def tokenise(text):
    """lower-case, one word = one token, add <START>/<END>."""
    return [[START] + line.lower().split() + [END] for line in text.strip().splitlines()]


class FirstOrderLM:
    def __init__(self, sentences):
        self.counts = defaultdict(Counter)          # counts[w_i][w_j] = C(w_i, w_j)
        for s in sentences:
            for prev, nxt in zip(s[:-1], s[1:]):
                self.counts[prev][nxt] += 1
        self.probs = self._build_probs()

    def _build_probs(self):
        # P(w_j | w_i) = C(w_i, w_j) / sum_k C(w_i, w_k)
        return {w: {v: c / sum(nx.values()) for v, c in nx.items()}
                for w, nx in self.counts.items()}

    def distribution(self, prev):
        return self.probs.get(prev, {})             # {} if context never seen

    def show(self, prev):
        d = self.distribution(prev)
        if not d:
            print(f"P(next | {prev}) : NO OBSERVED TRANSITIONS")
            return
        print(f"P(next | {prev}):")
        for w, p in sorted(d.items(), key=lambda x: -x[1]):
            print(f"    {w:<8s} {p:.4f}")

    def predict(self, prev):
        """argmax_w P(w | prev); ties broken by first-seen order (deterministic)."""
        d = self.distribution(prev)
        if not d:
            return None
        return max(d, key=d.get)

    def sample_next(self, prev):
        d = self.distribution(prev)
        if not d:
            return None
        words, ps = zip(*d.items())
        return random.choices(words, weights=ps, k=1)[0]

    def generate(self, mode="sample", max_len=20):
        """mode='sample' (Mode B) or 'greedy' (Mode A). Stops at <END> or max_len."""
        out, prev = [], START
        for _ in range(max_len):
            nxt = self.predict(prev) if mode == "greedy" else self.sample_next(prev)
            if nxt is None or nxt == END:
                break
            out.append(nxt)
            prev = nxt
        return " ".join(out)


def check_normalisation(probs, tol=1e-9):
    ok = True
    for w in sorted(probs):
        total = sum(probs[w].values())
        flag = "OK" if abs(total - 1) < tol else "FAIL"
        ok &= flag == "OK"
        print(f"  {w:<8s} sum P(v|w) = {total:.6f}  {flag}")
    return ok


if __name__ == "__main__":
    random.seed(42)
    sents = tokenise(DATA)
    lm = FirstOrderLM(sents)

    print("=== Conditional probability tables ===")
    for w in [START, "the", "cat", "dog", "sat", "ran", "on", "to", "mat", "rug", "park"]:
        lm.show(w)

    print("\n=== Zero-probability transitions (words in vocab, per context) ===")
    vocab = sorted({t for s in sents for t in s} - {START})
    for w in ["the", "cat", "dog", "sat", "ran"]:
        zeros = [v for v in vocab if v not in lm.distribution(w)]
        print(f"  after '{w}': {zeros}")

    print("\n=== Normalisation test ===")
    print("All rows sum to 1:", check_normalisation(lm.probs))

    print("\n=== Next-word prediction (argmax) ===")
    for w in [START, "the", "cat", "dog", "sat", "ran", "on", "to"]:
        print(f"  argmax P(. | {w}) = {lm.predict(w)}")

    print("\n=== 20 sampled sentences ===")
    gen = [lm.generate("sample") for _ in range(20)]
    for g in gen:
        print("  ", g)
    open("generated_first_order_20.txt", "w").write("\n".join(gen) + "\n")

    print("\n=== Greedy (Mode A), 5 sentences ===")
    for _ in range(5):
        print("  ", lm.generate("greedy"))
    print("\n=== Sampling (Mode B), 5 sentences ===")
    for _ in range(5):
        print("  ", lm.generate("sample"))
