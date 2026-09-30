"""Part XIII: compare first- and second-order models."""
import random
from first_order_lm import FirstOrderLM, tokenise, DATA, START, END
from second_order_lm import SecondOrderLM

random.seed(1)
sents = tokenise(DATA)
train = {" ".join(s[1:-1]) for s in sents}
m1, m2 = FirstOrderLM(sents), SecondOrderLM(sents)
V = sorted({t for s in sents for t in s} - {START})          # includes <END>
Vctx = [v for v in V if v != END]                            # words that can be a context

p1 = sum(len(v) for v in m1.probs.values())
p2 = m2.n_parameters()
ctx1_possible, ctx1_seen = len(Vctx) + 1, len(m1.probs)      # +1 for <START>
ctx2_possible, ctx2_seen = (len(Vctx) + 1) ** 2, len(m2.probs)
print(f"vocabulary (incl <END>): {len(V)}")
print(f"nonzero parameters : 1st={p1}  2nd={p2}")
print(f"contexts observed  : 1st={ctx1_seen}/{ctx1_possible}  2nd={ctx2_seen}/{ctx2_possible}")
print(f"zero-prob contexts : 1st={ctx1_possible-ctx1_seen}  2nd={ctx2_possible-ctx2_seen}")
print(f"full-table size (|ctx| x |V|): 1st={ctx1_possible*len(V)}  2nd={ctx2_possible*len(V)}")
print(f"training sentences (unique): {len(train)}")
for name, m in [("1st", m1), ("2nd", m2)]:
    g = [m.generate("sample") for _ in range(1000)]
    uniq = set(g)
    new = uniq - train
    print(f"{name}-order, 1000 samples: unique={len(uniq)}, unseen-in-training={len(new)}")
    for s in sorted(new)[:6]:
        print("     novel:", s)
