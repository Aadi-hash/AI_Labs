# AI Laboratory: Bayesian Networks and Autoregressive Language Models
**Lab record: answers to Questions 1–14, results, and reflection**

Files: `first_order_lm.py`, `second_order_lm.py`, `compare_models.py`, `output_first_order.txt`, `output_second_order.txt`, `output_comparison.txt`, `generated_first_order_20.txt`, `generated_second_order_20.txt`.
Run: `python3 first_order_lm.py`, `python3 second_order_lm.py`, `python3 compare_models.py`.

---

## Part I. Chain rule

**Q1. Why is the autoregressive decomposition useful for generating text?**
The chain rule turns one huge joint distribution P(X1..XT) into a product of small next-token distributions P(Xt | X1..Xt-1). This is exact, with no independence assumption. It gives a natural generation procedure: sample X1 from P(X1), then X2 from P(X2 | X1), and so on, each new token being appended to the context. It also means the same model does prediction (argmax of the next-token distribution), scoring (multiply the conditionals to get the probability of any sentence) and generation, and each conditional can be learned separately.

## Part II. Bayesian network

**Q2. Independence assumption of X1 → X2 → X3 → X4**
Each word depends only on its immediate predecessor (first-order Markov assumption):

P(Xt | X1, …, Xt-1) = P(Xt | Xt-1), equivalently Xt ⫫ {X1..Xt-2} | Xt-1.

The past and the future are conditionally independent given the present word.

## Part III/IV. Dataset and CPT

Tokens are lower-cased, and each sentence is wrapped as `<START> … <END>`. Vocabulary: the, cat, dog, sat, ran, on, to, mat, rug, park, `<END>` (plus `<START>`).

**Q3. Conditional probability tables, P(next | current)** (counts in the data → probabilities; from `output_first_order.txt`)

| current | next word : probability |
|---|---|
| `<START>` | the 1.0 |
| **the** (12 occurrences as a context) | cat 3/12 = 0.250, dog 3/12 = 0.250, mat 2/12 = 0.167, rug 2/12 = 0.167, park 2/12 = 0.167 |
| **cat** (3) | sat 2/3 = 0.667, ran 1/3 = 0.333 |
| **dog** (3) | sat 2/3 = 0.667, ran 1/3 = 0.333 |
| **sat** (4) | on 1.0 |
| **ran** (2) | to 1.0 |
| on | the 1.0 |
| to | the 1.0 |
| mat / rug / park | `<END>` 1.0 |

(The number of times each word is a context: `the` 12, `cat` 3, `dog` 3, `sat` 4, `ran` 2. The question's "cat three times" example is only illustrative.)

**Zero-probability transitions** (all other words in the vocabulary):
- after *the*: `<END>`, on, ran, sat, the, to
- after *cat* and after *dog*: everything except sat and ran (e.g. "cat cat", "cat mat", "dog park")
- after *sat*: everything except on (e.g. "sat ran", "sat the")
- after *ran*: everything except to

Most of the table is zero, so the model can never generate e.g. "the sat" or "cat park".

## Part V/VI. LLM implementation and inspection

The behavioural-specification prompt from the handout was used (spec: counts → P(Xt|Xt-1) → display → argmax → sampling → stop at `<END>`; no ML libraries). The result is `first_order_lm.py`.

**Q4. Where are the transition counts stored?** In `FirstOrderLM.counts`, a `defaultdict(Counter)`. `counts[prev][next]` is C(prev, next). It is filled in `__init__` by looping over `zip(s[:-1], s[1:])` for each sentence.

**Q5. Where is P(Xt | Xt-1) computed?** In `_build_probs()`. Each count is divided by the row total, `c / sum(nx.values())`, which is exactly C(wi,wj)/Σk C(wi,wk). The result is stored in `self.probs`.

**Q6. How does the program choose the next word?** There are two functions. `predict()` always picks the most probable word (greedy, `max(d, key=d.get)`), and `sample_next()` uses `random.choices(words, weights=probs)`. `generate(mode=…)` chooses between them, and the default is sampling. Difference: greedy is deterministic, always returns the mode of the distribution, and gives the same sentence every time. Sampling draws a word with probability equal to its model probability, so a word with p = 0.17 is chosen 17% of the time. The samples therefore follow the model's distribution and produce variety.

**Q7. What if a word has no observed transition?** `distribution()` returns `{}` (via `.get(prev, {})`), `predict` and `sample_next` return `None`, and `generate` then stops the sentence cleanly instead of raising a `KeyError`. In the model itself the probability is undefined (0/0), since the data gives no information. Fixes are smoothing (e.g. add-one), back-off to a shorter context, or an `<UNK>` token. Example in the second-order model: context (cat, park) → "NO OBSERVED TRANSITIONS".

## Part VII. Testing

**Normalisation test** (`check_normalisation`), first-order output, every row:
`<START> 1.000000, cat 1.000000, dog 1.000000, mat 1.000000, on 1.000000, park 1.000000, ran 1.000000, rug 1.000000, sat 1.000000, the 1.000000, to 1.000000`, all OK.
The second-order model's 15 contexts also all sum to 1.000000 (see `output_second_order.txt`).

**Q8. If a total is 0.87?** The row is not a valid probability distribution, so the implementation is wrong. Probability mass (0.13) has been lost. Likely causes: dividing by the wrong denominator (e.g. total count of all tokens instead of the row total), skipping/not counting some transitions (e.g. `<END>` dropped, or the last token of each sentence missed), an off-by-one in the pairing loop, or accidental smoothing/pruning without renormalising. The model then no longer represents a probability distribution and sampling would be biased.

## Part VIII. Prediction

argmax P(w | current) (`output_first_order.txt`):

| context | argmax | P |
|---|---|---|
| `<START>` | the | 1.00 |
| the | cat (tie with dog, broken by first-seen order) | 0.25 |
| cat | sat | 0.667 |
| dog | sat | 0.667 |
| sat | on | 1.00 |
| ran | to | 1.00 |
| on | the | 1.00 |
| to | the | 1.00 |

Full distribution for "the": cat .25, dog .25, mat .167, rug .167, park .167.

**Q9. Do predictions match human expectations?** Only partly. After "sat", "on" is what a human expects, but after "the" the model's "most probable" word is a coin-flip between cat and dog (a pure tie, so the pick is arbitrary), whereas a human uses meaning and context: after "the cat sat on the" they would expect mat. The model only knows the counts of its tiny training set. It knows nothing about grammar, semantics, or the world, and it cannot see beyond one word. A probability model reflects the statistics of its training data and its structural assumptions. Human expectation reflects knowledge, meaning and much longer context, so the two agree only when the data and context are rich enough.

## Part IX. Generation

20 sampled sentences from the first-order model (seed 42, also in `generated_first_order_20.txt`) include:
"the dog sat on the mat", "the park", "the mat", "the rug", "the cat sat on the cat sat on the mat", "the dog ran to the dog sat on the park", and a 20-token run "the cat sat on the dog ran to the cat sat on the cat sat on the dog ran to" that was cut off by `max_len`.
Several are ungrammatical ("the park", "the mat") because the model cannot remember the earlier context.

## Part X. Greedy vs sampling

Greedy (5 sentences): all identical, and *never terminating*: `the cat sat on the cat sat on the cat sat on …` (cut off by `max_len = 20`). The loop the → cat → sat → on → the → cat … has a cycle in the argmax path that never reaches `<END>`.
Sampling (5): "the dog sat on the dog sat on the cat ran to the mat", "the park", "the cat ran to the rug", "the park", "the rug".

**Q10. Which mode has more variation, and why?** Sampling. Greedy is deterministic (always the mode), so all five sentences are identical. Sampling follows the full distribution, so alternative words (dog, mat, rug, park, ran) are drawn with their probabilities, and the same run gives different outputs. Greedy can also fall into loops because the most likely path may never reach `<END>`. (A `max_len` guard is therefore needed.)

## Part XI/XII. Second-order model

Implementation: `second_order_lm.py`. Counts of triples `counts[(a, b)][c]`; sentences are padded with two `<START>` tokens so X1 ~ P(·|`<START>`,`<START>`). Sampling, greedy, normalisation test and unseen-context handling work as in the first-order model. Selected CPTs:

| context (Xt-2, Xt-1) | distribution |
|---|---|
| (`<START>`,`<START>`) | the 1.0 |
| (`<START>`, the) | cat 0.5, dog 0.5 |
| (the, cat) / (the, dog) | sat 0.667, ran 0.333 |
| (cat, sat), (dog, sat) | on 1.0 |
| (sat, on) | the 1.0 |
| (on, the) | mat 0.5, rug 0.5 |
| (ran, to) | the 1.0 |
| (to, the) | park 1.0 |
| (cat, park) | unseen, no data |

**Q11. First- vs second-order**
1. **Graph structure:** first-order has one parent per node (Xt-1 → Xt). Second-order has two parents (Xt-2 → Xt ← Xt-1), so Xt is a collider, with additional edges.
2. **CPT:** the first-order CPT is indexed by one word, |V| rows × |V| columns. The second-order CPT is indexed by pairs, |V|² rows × |V| columns, so it is a much larger table (here 121 vs 1331 possible entries).
3. **Context:** first-order sees 1 previous word, second-order sees 2. E.g. after "the" the first-order model cannot tell cat from dog, but (on, the) vs (to, the) already distinguishes {mat, rug} from {park}.
4. **Data needed:** far more. The number of contexts grows as |V|², and each context needs enough observations to estimate its row. Most contexts are never seen (106 of 121 here).

## Part XIII. Comparison (`output_comparison.txt`)

| measure | first-order | second-order |
|---|---|---|
| distinct non-zero parameters | 17 | 19 |
| contexts observed / possible | 11 / 11 | 15 / 121 |
| zero-probability (unseen) contexts | 0 | 106 |
| full CPT size (contexts × 11 words) | 121 | 1331 |
| unique sentences in 1000 samples | 171 | 6 |
| samples that are not in the training set | 165 | 0 |
| coherence | often ungrammatical or looping ("the park", "the cat ran to the cat ran to the cat ran to the park") | every sample is a well-formed sentence, but every one is one of the 6 training sentences |

Example sentences: 1st order: "the dog ran to the dog sat on the park". 2nd order: "the cat ran to the park", "the dog sat on the rug". Greedy 2nd-order always gives "the cat sat on the mat".
The second-order model is coherent here only because it has effectively memorised the corpus (each of its contexts has almost only one continuation). It is not more "creative", it is less. The first-order model is more diverse only because its weaker assumption lets it produce many combinations that are mostly nonsense.

**Q12. Why can more context improve prediction but make estimation harder?** More context removes ambiguity: P(Xt | Xt-2, Xt-1) is closer to the true P(Xt | X1..Xt-1), so predictions are sharper and more accurate (e.g. "on the" → mat/rug but "to the" → park). But the CPT has |V|^k rows for context length k, and it grows exponentially. With a fixed amount of data each row is estimated from fewer examples, many rows have zero counts (here 106 of 121 contexts are unseen), and the observed rows are noisy or memorised. This is the bias–variance trade-off: a small context has high bias but low variance, a large context has low bias but high variance. That is why smoothing/back-off are needed, and why neural networks, which share parameters across contexts, are used instead of explicit tables.

## Part XIV/XV. Modern models and the role of the LLM

The neural network replaces the CPT: it takes x1..xt-1 and outputs P(Xt | x1..xt-1) through learned weights, trained by gradient descent rather than counting. The chain-rule objective P(x1..xT) = Π P(xt | x1..xt-1), with the first term P(x1 | `<START>`), is the same as in the lab.

**Q13. Why is Approach B preferable?**
- **Specifying intended behaviour:** B states the model (variables, conditional, estimation, generation), so the LLM implements a defined thing. A leaves it free to produce any "language model", possibly a neural net, or an n-gram with smoothing, unrelated to the course.
- **Understanding the representation:** with B I know the state to look for (a count table, row-normalised into a CPT), so I can locate and read it in the code.
- **Validating the implementation:** the spec is a checklist (do the counts exist? is the CPT computed as count/row-total? does generation stop at `<END>`?), so correctness can be judged against something.
- **Testing probabilistic invariants:** B gives testable properties, e.g. each row sums to 1, probabilities ≥ 0, zero counts give zero probability, sampling frequencies converge to the CPT. With A there is nothing precise to test.
- **Implementation vs model:** the model is the mathematical object P(Xt | Xt-1). The code is one implementation of it, and a program that runs and prints plausible text may still not implement that model (e.g. wrong denominator, sampling uniformly). B lets me separate the two and check that they match.

**Q14. What did the Bayesian network give me?** (five of the listed points)
1. **Representation of dependencies:** the graph makes explicit which variables Xt depends on (Xt-1, or Xt-2 and Xt-1), so the structure is visible before any code is written.
2. **Factorisation of the joint:** P(X1..XT) = Π P(Xt | parents(Xt)), so one large distribution reduces to small local CPTs that can be estimated from counts.
3. **Independence assumptions:** the missing edges state the assumption Xt ⫫ X1..Xt-2 | Xt-1 precisely. It also shows why the first-order model can't tell "the cat sat on the" from "the dog ran to the".
4. **Principled generation:** ancestral sampling, in topological order, from each CPT gives exact samples from the model. Greedy decoding, by contrast, is not sampling from it (and got stuck in a loop).
5. **Effect of increasing context:** adding parent edges shows the CPT growing exponentially (121 → 1331 entries), which explains sparsity, memorisation and the data problem.
6. **Testing:** the probabilistic specification gives invariants (rows sum to 1, zero counts ↔ zero probability) that can be checked automatically against the code.

## Deliverable 7. Reflection on LLM use

*(Adapt this to your own session. The examples below are real issues that the tests exposed in this implementation.)*

I gave the LLM the behavioural specification from Part V and then read the generated code against the model. I checked where counts were stored (`counts[prev][next]`), that P was computed as count / row total, that generation samples with `random.choices` and that greedy uses `max`, and I ran the normalisation test (all rows 1.0).

Issues I found while inspecting and testing, and corrected:
1. **Greedy generation never terminates.** The argmax path the → cat → sat → on → the → cat … never reaches `<END>`, so a `while True` loop runs forever. I added `max_len` to `generate()`.
2. **Unseen context.** A naive `self.probs[prev]` raises `KeyError`. I changed it to `.get(prev, {})` and made `generate` stop when there is no distribution. In the second-order model, (cat, park) is such a context.
3. **Tie-breaking.** For "the", cat and dog are tied at 0.25, so argmax is decided by dict insertion order. It is deterministic in Python 3.7+, but it is arbitrary, and I noted this in my answer to Q9.
4. **Second-order start-up.** Two `<START>` tokens are needed so that the first two words also have a full context.

Validation was by: reading the code against the spec, the normalisation invariant, comparing printed CPTs to hand-computed counts (e.g. the → cat = 3/12), and checking that 1000 sampled sentences from the second-order model are all valid training sentences (consistent with its CPT).
