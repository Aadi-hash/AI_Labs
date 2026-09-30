"""Second-order autoregressive model: P(X_t | X_{t-2}, X_{t-1}) from counts of observed triples.
Structure: X_{t-2} -> X_t <- X_{t-1}.  Sentences are padded with two <START> tokens so that
X1 ~ P(. | <START>,<START>) and X2 ~ P(. | <START>, X1)."""
import random
from collections import defaultdict, Counter
from first_order_lm import START, END, DATA, tokenise


class SecondOrderLM:
    def __init__(self, sentences):
        self.counts = defaultdict(Counter)          # counts[(a,b)][c] = C(a,b,c)
        for s in sentences:
            s = [START] + s                          # second <START> pad
            for a, b, c in zip(s[:-2], s[1:-1], s[2:]):
                self.counts[(a, b)][c] += 1
        self.probs = {ctx: {w: c / sum(nx.values()) for w, c in nx.items()}
                      for ctx, nx in self.counts.items()}

    def distribution(self, a, b):
        return self.probs.get((a, b), {})

    def show(self, a, b):
        d = self.distribution(a, b)
        print(f"P(next | {a}, {b}):" + ("" if d else "  NO OBSERVED TRANSITIONS"))
        for w, p in sorted(d.items(), key=lambda x: -x[1]):
            print(f"    {w:<8s} {p:.4f}")

    def predict(self, a, b):
        d = self.distribution(a, b)
        return max(d, key=d.get) if d else None

    def sample_next(self, a, b):
        d = self.distribution(a, b)
        if not d:
            return None
        w, p = zip(*d.items())
        return random.choices(w, weights=p, k=1)[0]

    def generate(self, mode="sample", max_len=20):
        out, a, b = [], START, START
        for _ in range(max_len):
            nxt = self.predict(a, b) if mode == "greedy" else self.sample_next(a, b)
            if nxt is None or nxt == END:
                break
            out.append(nxt)
            a, b = b, nxt
        return " ".join(out)

    def n_parameters(self):
        return sum(len(v) for v in self.probs.values())


def check_normalisation(probs, tol=1e-9):
    ok = True
    for ctx in sorted(probs):
        total = sum(probs[ctx].values())
        flag = "OK" if abs(total - 1) < tol else "FAIL"
        ok &= flag == "OK"
        print(f"  {str(ctx):<28s} sum = {total:.6f}  {flag}")
    return ok


if __name__ == "__main__":
    random.seed(42)
    sents = tokenise(DATA)
    lm = SecondOrderLM(sents)

    print("=== Selected CPTs ===")
    for ctx in [(START, START), (START, "the"), ("the", "cat"), ("the", "dog"),
                ("cat", "sat"), ("dog", "sat"), ("sat", "on"), ("on", "the"),
                ("ran", "to"), ("to", "the")]:
        lm.show(*ctx)

    print("\n=== Normalisation test ===")
    print("All contexts sum to 1:", check_normalisation(lm.probs))

    print("\n=== Unseen context example ===")
    lm.show("cat", "park")

    print("\n=== 20 sampled sentences ===")
    gen = [lm.generate("sample") for _ in range(20)]
    for g in gen:
        print("  ", g)
    open("generated_second_order_20.txt", "w").write("\n".join(gen) + "\n")

    print("\n=== Greedy (5) ===")
    for _ in range(5):
        print("  ", lm.generate("greedy"))
    print("=== Sampling (5) ===")
    for _ in range(5):
        print("  ", lm.generate("sample"))
