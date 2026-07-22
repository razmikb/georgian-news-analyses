# The eval set — how to label it

`pairs_to_label.csv` is 60 pairs of real headlines from our own database. Your job is to
say, for each pair, whether the two headlines are about **the same event** or not.

## Why this matters

Clustering comes down to one number: how similar two articles must be before the pipeline
calls them the same story. Too loose and unrelated stories get merged; too tight and one
event splits into three — which quietly breaks blindspot detection, because a story looks
one-sided only when its other half has landed in a separate cluster.

Your labels are how that number gets *measured* instead of guessed. Everything downstream
— coverage bars, blindspots — is only as trustworthy as this file.

## How to do it

1. Open `pairs_to_label.csv` in Excel or Google Sheets. (It's saved so Georgian text opens
   correctly; if it ever looks like `áƒ¡áƒáƒ¥`, tell Claude rather than fixing it by hand.)
2. Read column **headline_a** and **headline_b**. Fill in column **label** with one of:

   | label | means |
   |---|---|
   | `same` | Same real-world event. A reader who saw both would say "I've read this already." |
   | `different` | Different events, even if the topic or people overlap. |
   | `unsure` | Genuinely ambiguous even after opening both. Better than a coin flip — see below. |

   **Judge the two articles, not the two headlines.** If a headline is too vague to place,
   open its link (`url_a` / `url_b`) and look. Most pairs are obvious from the headline
   alone, so this should be a handful of rows, not all sixty — but where it matters, look.

3. If a call was hard or the reason isn't obvious, add a few words in **notes**. Those notes
   are worth a lot when we later look at which pairs the pipeline got wrong.
4. Save as CSV (keep the same filename) and tell Claude it's ready.

`unsure` is a real answer, not a failure. Those pairs are exactly the ones the pipeline
will send to a second check (PLAN.md §6), so knowing which cases are ambiguous *to a human*
is useful information — don't force them into same/different. But reserve it for pairs that
stay ambiguous after you've opened both links, not ones where the headline was just thin.

## Why the article text isn't in this file

The pipeline will not cluster on headlines alone — it embeds **headline + lead paragraphs**
(PLAN.md §6). So why does this sheet show only headlines?

Because this file is the *answer key*, not a picture of what the pipeline sees. Your label
records what is true about the two articles; the pipeline's job is to reproduce your answers
from whatever text we feed it. Those are deliberately separate. In fact one of the things
we'll tune on this set is **how much text to use** — headline only vs. headline + lead vs.
full body. That comparison is only possible if the answers were reached with full knowledge
of the articles. Label from the headlines alone and the key inherits their vagueness, so
"adding the lead paragraph helped" becomes unmeasurable.

The lead paragraphs are not pasted into the sheet for a second reason: we store headlines
and links only, never article text (PLAN.md §9). The pipeline may read a body in memory to
compute an embedding, but a CSV committed to the repo is storage. Hence links.

## Judgment calls

The rule is **one event**, not one topic or one story arc.

- Prosecution *requests* bail vs. the court *grants* bail — same incident, but arguably two
  events on the same day. Your call decides how the product behaves; there's no
  right answer waiting to be looked up. Be consistent and note what you decided.
- Two separate car crashes, two separate court hearings, two strikes in different wars —
  `different`, however alike the wording.
- A story and its follow-up interview about that story — your call; note it.
- If a headline is too vague to place ("What the woman involved says") but the other one
  makes the event clear, judge whether a reader would take them as the same story.

Consistency matters more than any individual verdict. If you change your mind about a rule
halfway through, say so — we can revisit the earlier rows.

## Columns you can ignore

`band` and `word_overlap` describe how the pair was *chosen*, not what it is. `band` is not
a hint at the answer — the "similar" band deliberately contains lookalikes that are
different events, and the "unrelated" band should be almost all `different` (those are the
control rows that catch a threshold set far too loose). `url_a` / `url_b` are there for
when the headlines alone aren't enough and you want to open the articles.

## Where these 60 came from

Picked by `pipeline/eval_pairs.py` from the ~350 headlines in Supabase. Random pairs would
have been ~99% "different" and would have taught us nothing about where the boundary sits,
so the sample is **stratified**: pairs are grouped by how much wording they share, and each
group is sampled separately. Consequence to remember later: this file is *not* a natural
sample of Georgian news. It is deliberately dense with hard cases, so "we got 90% right on
the eval set" does not mean "90% right in production" — it's a tuning instrument, not a
report card.

To regenerate (this overwrites any labels — don't run it once labelling has started):

```
python -m pipeline.eval_pairs
```
