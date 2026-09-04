# cm-benchmark — Evaluation Process Manual

**Scope:** How we decide an item (question + answer + distractors + `answer_source`) is *correct and well-built* before it can enter the FREEZE'd, VLM-evaluated set. This manual governs the **Item Generation Pipeline** (`generator/` → `generation/` → FREEZE), not the downstream Model Evaluation Pipeline.

**Non-goal:** This is not a rubric for grading VLM answers. Item correctness and model scoring are separate concerns (invariant #3: scoring is deterministic code, no LLM judge).

---

## 1. What "correct" means here

An item is correct only if it passes **all four** of these, not just "the answer looks right":

| # | Property | Source of truth |
|---|---|---|
| 1 | **Geometrically true** — the locked `answer` matches what the metadata actually says | Recomputation from poses/edges/tracks (deterministic code) |
| 2 | **Non-leaking** — the question text does not state the answer, a direction, or a move/event | `shared_rules` in `taxonomy.yaml` |
| 3 | **Vision-necessary** — cannot be solved by a blind-LLM or caption→LLM baseline | invariant #4 |
| 4 | **Construct-faithful** — the item actually tests the construct it's tagged with, per that construct's `discriminators` | `taxonomy.yaml` per-construct block |

Properties 1–2 are things code can check deterministically. Property 3 needs a baseline run (blind/caption LLM) plus optional human confirmation. Property 4 is where human judgment matters most — it's easy for a generator bug to produce an item that is *factually correct* but tests the wrong thing (e.g., an `egocentric_encoding` item where the object is secretly also inferable allocentrically, or a `spatial_updating` item where the object was still visible at the final pose, silently turning it into class-1 encoding).

This is why the process has two layers: **automated validation** (cheap, exhaustive, catches 1–2) and **human review** (expensive, sampled, catches 3–4 and anything the automated layer can't see, like whether the image sequence is genuinely legible).

---

## 2. Layer A — Automated / deterministic validation

Runs on every candidate item before it's eligible for human review. This is the `GT Validator` + `vision-necessity` step on your roadmap; treat this manual as its spec.

### 2.1 Structural checks (schema-level, per item)
- `answer_source` is non-null and resolves to a real field/row in the episode store (invariant #2).
- `answer` is one of `options`; exactly one option is correct.
- `frame_of_reference` is set and matches what the construct requires (fixed per construct, or explicit in multi-frame items).
- For temporal constructs (class 2–4): question text situates “now” / “{k} steps ago”
  relative to the live navigation stream (online sequential protocol). Do **not** require
  a bundled multi-image “time order” cue. Flag as bug if temporal reference is ambiguous.
- `image_paths` are non-empty and resolvable to real files.
- No duplicate `item_id`.

### 2.2 Answer recomputation (per construct)
Re-derive the answer independently from the episode DB and compare to the stored `answer`:

- `egocentric_encoding` → recompute viewer→object direction from `agent_pose@shown_step` + object position.
- `allocentric_encoding` → recompute object→object relation using the reference object's intrinsic frame.
- `spatial_working_memory` → recompute recalled relation/location/count at the stated `k` steps back; confirm object was static across shown frames.
- `invisible_displacement` → confirm `displacement_event.hidden_during == true` and object is not visible at its true final location in the last frame; recompute final container.
- `spatial_updating` → confirm ≥1 real movement action between encode and query steps; confirm object position unchanged; recompute bearing from `agent_pose@final`.
- `perspective_taking` → confirm `reference_entity_facing_heading` exists and is used (not the camera frame); recompute relation in that frame. **Currently `status: unsupported`** — do not promote items for this construct until facing metadata is trusted.
- `route_knowledge` → confirm the queried path is actually present in `agent_trajectory` (was traversed); recompute turn order.
- `survey_based_route_planning` → confirm the queried path/shortcut is **absent** from `agent_trajectory` (never traversed) and is derivable from `world_layout.connectivity`; reject if it reduces to an experienced route.

Mismatch between recomputed and stored answer = automatic reject, routed back to the generator/template, not to human review.

### 2.3 Distractor diagnosticity check
Each construct declares a `distractor_pattern` (e.g., `[opposite_direction, orthogonal_direction, plausible_wrong_axis]`). For every item, confirm:
- each distractor maps to a named failure mode in that construct's pattern (not a generic random option),
- `distractor_rationale` names which failure mode each wrong option encodes,
- no distractor is accidentally also geometrically correct (recompute each distractor the same way as the answer — this catches a common generator bug where two options tie).

### 2.4 Vision-necessity check (baseline gate)
Run two baselines on the candidate item:
- **Blind-LLM**: question + options only, no images.
- **Caption→LLM**: question + options + an off-the-shelf caption of the image(s) (no direct pixel access).

If either baseline gets the item right at above-chance rate across repeated samples (e.g., >20% above the 1/N option-count floor), the item is **not** vision-necessary and is rejected or sent back for harder distractors. This operationalizes invariant #4 and is the automatable half of Layer B's "is this really testing vision" question — human review below adds the qualitative half (*is the image actually legible to a person*, which a text baseline can't tell you).

**Output of Layer A:** each candidate gets `status ∈ {auto_pass, auto_fail}` plus a machine-readable `fail_reason` list. Only `auto_pass` items proceed to Layer B.

---

## 3. Layer B — Human review

This is the part you flagged as necessary because some judgments (construct-faithfulness, image legibility, question naturalness, whether a "plausible but unwalked route" distractor is actually plausible to a human) aren't fully checkable by code, and because Layer A's baselines can be gamed by items that are accidentally easy in ways code doesn't anticipate.

### 3.1 Who reviews, and with what
- **Annotators:** 2 independent reviewers minimum per sampled item. Reviewers should be spatial-reasoning-literate but *not* the person who wrote the generator/template for that construct (avoid confirmation bias — you already do this pattern with the visibility labeling tool being separate from generation code).
- **Tooling:** extend the existing `build_labeling_set.py` / annotated-frame pattern. Reuse `annotate_frames.py` output (numbered circles + legend) as the visual the reviewer sees — this keeps review artifacts consistent with what you already build for visibility calibration, and means you don't need a second annotation UI.
- **Blinding:** reviewers see item images + question + options, but **not** the stored answer or `answer_source`, until after they submit their own judgment. This makes the human review a genuine second measurement, not a rubber stamp.

### 3.2 What each reviewer judges, per item
A checklist derived directly from the construct's `discriminators` in `taxonomy.yaml` — this is the key move: **the taxonomy already tells you the rubric**, you don't need to invent a separate one. For example, for `spatial_updating` the reviewer checklist is literally:
- [ ] Object position is unchanged across the shown frames (only the agent moved)
- [ ] The agent visibly executes ≥1 real movement action between encode and query
- [ ] The queried object is not visible at the final pose
- [ ] The question is answerable from the shown frames + geometry alone, not world knowledge

Plus three checks that apply to every construct (from `shared_rules`):
- [ ] Question does not name the answer, a direction, or a move/event
- [ ] Exactly one option is defensibly correct given only what's shown
- [ ] Frame of reference is unambiguous

Plus one holistic item:
- [ ] **Can you personally answer this correctly from the images alone**, and would you expect a spatially competent adult to agree? (This is the human analogue of the blind-LLM baseline — catches cases where the "vision-necessary" gate passed but the image is actually ambiguous, blurry, or the object is technically-but-not-practically visible.)

Reviewer selects an option (their own answer to the question) *before* unblinding, then marks each checklist box, then is shown the stored answer/rationale and confirms or files a discrepancy.

### 3.3 Decision rule
- **Agree + checklist clean, both reviewers:** item promoted, `status: verified`.
- **Disagreement between reviewers** (different chosen answers, or one flags a discriminator failure): route to a third/adjudicating reviewer (should be you, or whoever owns the taxonomy) who makes the final call and — importantly — records *which discriminator or shared rule was violated*, so it maps back to a template bug, not just "reject."
- **Both reviewers reject on the same discriminator:** item rejected, and the underlying template/generator gets a bug ticket referencing the specific `discriminators` line that failed. Don't just discard the item silently — a systematic discriminator failure across many items means the generator itself is broken for that construct.

### 3.4 Inter-annotator agreement (IAA)
Track agreement per construct, not just globally — constructs differ hugely in difficulty (`perspective_taking` and `survey_based_route_planning` are inherently harder to adjudicate than `egocentric_encoding`). Report:
- Percent agreement and Cohen's κ per construct, computed on the "selected option" field (this is the harder, more informative metric — checklist agreement alone can look good even when reviewers are answering the underlying question differently).
- A construct with κ below your threshold (pick one — 0.6 is a common floor for "substantial agreement" in annotation literature, but you should set this deliberately rather than import it blindly) means the construct's items are ambiguous to humans too, and the fix is almost always in `question_template` wording or `distractor_pattern`, not in adding more reviewers.

### 3.5 Sampling
At this stage of the project (small draft sets, active template iteration), review **100% of `auto_pass` items** — you don't have volume yet where sampling saves meaningful effort, and full review gives you the richest signal for fixing templates. Once templates stabilize and volume grows, move to a stratified sample (e.g., all items from any construct/scene combo not yet seen, plus a fixed percentage of repeats) and hold out a fully-reviewed set as a periodic audit.

---

## 4. Feedback loop back into generation

Every rejection (Layer A or B) should carry a **reason code** that maps to one of:
- a specific `discriminators` line (construct-faithfulness bug),
- a `shared_rules` line (leakage / ambiguity bug),
- a `distractor_pattern` entry (weak distractor bug),
- a metadata/data gap (e.g., missing `reference_entity_facing_heading` — expected right now for `perspective_taking`).

Aggregate reason codes per construct per generator version. This turns human review from a one-off gate into the mechanism that tells you *which template to fix next*, which is more useful to you right now (early, iterating on `templates.py`/`constructs.py`) than a single pass/fail number.

## 5. Gate to FREEZE

An item is eligible for the immutable frozen set only when:
1. Layer A: `auto_pass` on all checks in §2, including vision-necessity baselines, and
2. Layer B: `status: verified` (or adjudicated to verified) with no open discrepancy, and
3. `vision_necessary == true` is recorded, satisfying invariant #4.

`perspective_taking` items should not be promoted to FREEZE while `status: unsupported` — Layer B review can still run on them to unblock the metadata work, but treat any current draft output for this construct as *template debugging*, not benchmark content.

## 6. Open decisions for you to set (not code-decidable)

- IAA threshold per construct (§3.4).
- Whether Layer B is 100% or sampled, and when to switch (§3.5).
- Who adjudicates disagreements (§3.3) — needs to be a fixed, named role, not "whoever's free."
- Baseline models/thresholds for the vision-necessity gate (§2.4) — which blind-LLM and captioner, and what "above chance" margin you'll accept.