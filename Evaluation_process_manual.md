# cm-benchmark — Evaluation Process Manual

**Scope:** How we decide an item is well-built before FREEZE. This is the Item
Generation Pipeline, not model scoring. Class-4 scoring details:
[`docs/generation.md`](docs/generation.md).

**Non-goal:** This is not a rubric for grading VLM answers. Item correctness and model scoring are separate concerns (invariant #3: scoring is deterministic code, no LLM judge). For a manual VLM chat/tool pilot (attach `prompt.txt` + frames, paste the kit upload message, score with `analyze_review.py --vlm-kit`), see **VLM chat kit** in [`docs/generation.md`](docs/generation.md).

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
- For MCQ constructs, `answer` is one of `options`; exactly one option is correct. Class-4 (`route_knowledge`, `survey_based_route_planning`) has empty `options`: the model writes a free action sequence; [CODE] scoring is metric simulation (success / validity / SPL), not option match. The allowed action names are stated once in `cm_benchmark.evaluation.protocol.SYSTEM_INSTRUCTION` (the Model Runner wraps every item with it) — not in the per-item question.
- `frame_of_reference` is set and matches what the construct requires (fixed per construct, or explicit in multi-frame items).
- For temporal constructs (class 2–4): question text situates “now” / “{k} steps ago”
  relative to the live navigation stream (online sequential protocol). Do **not** require
  a bundled multi-image “time order” cue. Flag as bug if temporal reference is ambiguous.
- `image_paths` are non-empty and resolvable to real files. When
  `disambiguation` is `mark`, these are the colour-dot copies (not the
  unmarked collection frames). Reviewers must judge the marked frames
  the model will see. The phrase in the question (`marked with a
  {colour} dot`) must match the dots on those copies.
- No duplicate `item_id`.

### 2.2 Answer recomputation (per construct)
Re-derive the answer independently from the episode DB and compare to the stored `answer`:

- `egocentric_encoding` → recompute viewer→object direction from `agent_pose@shown_step` + object position.
- `allocentric_encoding` → recompute object→object relation using the reference object's intrinsic frame.
- `spatial_working_memory` → recompute recalled relation/location/count at the stated `k` steps back; confirm object was static across shown frames. `image_paths` must be every FOV frame from encode through query in time order (turns in place stay).
- `invisible_displacement` → confirm the destination leaves view within `invisible_displacement.max_delay`, the object / old pose / destination stay out of view until then, and a clear pre-query sighting of the destination (or swap partner / floor landmark) exists; recompute ego bearing at the query pose. Destination identity is exact `obj-id`. For Floor destinations, confirm a distinguishable floor-anchor landmark within `FLOOR_ANCHOR_RADIUS` of the true final position (or reject that candidate).
- `spatial_updating` → confirm **net pose change** (position *or* heading delta above tolerance) between encode and query — not action count alone; confirm object static via `object_state_track`; recompute bearing from `agent_pose@final` + object position; drop duplicate (object, encode) items with identical answers. `image_paths` must be every FOV frame from encode through query in time order.
- `perspective_taking` → confirm A, B, and C are distinguishable together in the one shown frame; recompute `imagined_perspective_label(A, B, C)` (signed A→B vs A→C angle → left/right/behind; reject near 0°/±135°). Distractors: camera frame, mirrored L/R, wrong facing.
- `route_knowledge` → path exists on traversed_edges; score with `score_route_action_sequence` (success / validity / SPL). Frames are an even sample of the walk from source sighting minus `context_margin_steps` through goal sighting plus that margin, at most `class4.n_frames`, always including both sightings.
- `survey_based_route_planning` → no path on traversed_edges; a path on viewed_edges; score with `score_survey_action_sequence` (same simulator; validity requires at least one unwalked viewed edge). Frames are an even sample of the walk through the query, at most `class4.n_frames`, always including source/goal sightings and one frame per planned-path edge.

Mismatch between recomputed and stored answer = automatic reject, routed back to the generator/template, not to human review.

### 2.3 Distractor diagnosticity checks
Class-4 items have empty `options` / `distractor_pattern` (free action sequences). For every **MCQ** construct, confirm:
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
- **Annotators:** 2 independent reviewers minimum per sampled item (`--reviews-per-item`). Reviewers should be spatial-reasoning-literate but *not* the person who wrote the generator/template for that construct.
- **Tooling:** `scripts/build_review_package.py` writes one zip-ready folder per reviewer. Each folder has `index.html` and **exactly the frames in that item's `image_paths`** (marked copies when the item uses marks). Gold stays in `manifest.json` at the package root, not in the reviewer folders. Multi-frame items hide the question until **Show the question**. After review, `scripts/analyze_review.py` merges the downloaded CSVs against the manifest (accuracy, Krippendorff's α, flags). Commands: [`docs/generation.md`](docs/generation.md).
- **Blinding:** reviewers see the same frames the model gets, plus the question. For MCQ they pick an option (or Can't tell). For route / survey they answer only whether the route can be worked out from those frames (yes/no), with an optional free-text route. They do **not** see the stored gold until after they submit (practice items are the exception: they show the stored answer and are skipped by analysis).

### 3.2 What each reviewer judges, per item
A checklist derived from the construct's `discriminators` in `taxonomy.yaml`. The HTML flags are shared across constructs: ambiguous reference, unclear question, object not visible, more than one option looks correct.

For every construct (from `shared_rules`):
- [ ] Question does not name the answer, a direction, or a move/event
- [ ] Frame of reference is unambiguous
- [ ] The frames shown are enough to answer (for MCQ: exactly one option is defensibly correct; for route/survey: the route from A to B can be worked out from these frames)

Construct-specific checks that matter in review:
- [ ] **spatial_updating / spatial_working_memory:** the sequence is consecutive FOV frames for the stated `k` (no missing turn in the middle)
- [ ] **invisible_displacement:** the destination place is clearly visible in some shown frame before the query
- [ ] **perspective_taking:** A, B, and C appear together in the one shown frame
- [ ] **route / survey:** the shown walk is enough to work out A→B; do not grade a written action list as gold

Plus one holistic item:
- [ ] **Can you personally answer this from the images alone**, and would you expect a spatially competent adult to agree? (Catches cases where automated gates passed but the image is ambiguous, blurry, or the object is only technically visible.)

On MCQ items the reviewer selects an option *before* seeing gold. On route/survey items the scored judgment is yes/no on whether the route can be worked out; the optional written route is free text for qualitative notes, not the benchmark gold.

### 3.3 Decision rule
- **Agree + checklist clean, both reviewers:** item promoted, `status: verified`.
- **Disagreement between reviewers** (different chosen answers, or one flags a discriminator failure): route to a third/adjudicating reviewer (should be you, or whoever owns the taxonomy) who makes the final call and — importantly — records *which discriminator or shared rule was violated*, so it maps back to a template bug, not just "reject."
- **Both reviewers reject on the same discriminator:** item rejected, and the underlying template/generator gets a bug ticket referencing the specific `discriminators` line that failed. Don't just discard the item silently — a systematic discriminator failure across many items means the generator itself is broken for that construct.

### 3.4 Inter-annotator agreement (IAA)
Track agreement per construct, not just globally — constructs differ hugely in difficulty (`perspective_taking` and `survey_based_route_planning` are inherently harder to adjudicate than `egocentric_encoding`). `scripts/analyze_review.py` reports human accuracy vs gold, nominal Krippendorff's α, and flag rates on study items (practice rows are ignored). For MCQ, gold is the stored letter; for route/survey, gold for the yes/no judgment is `yes` (the question is answerable from the frames). A construct with α below your threshold means the items are ambiguous to humans too, and the fix is almost always in wording, frame selection, or distractors — not in adding more reviewers.

### 3.5 Sampling
At this stage of the project (small item sets, active template iteration), review **100% of `auto_pass` items** — you don't have volume yet where sampling saves meaningful effort, and full review gives you the richest signal for fixing templates. Once templates stabilize and volume grows, move to a stratified sample (e.g., all items from any construct/scene combo not yet seen, plus a fixed percentage of repeats) and hold out a fully-reviewed set as a periodic audit.

---

## 4. Feedback loop back into generation

Every rejection (Layer A or B) should carry a **reason code** that maps to one of:
- a specific `discriminators` line (construct-faithfulness bug),
- a `shared_rules` line (leakage / ambiguity bug),
- a `distractor_pattern` entry (weak distractor bug),
- a metadata/data gap (e.g., missing `nav_graph` for class-4, or no viewed-edge path for `survey_based_route_planning`).

Aggregate reason codes per construct per generator version. This turns human review from a one-off gate into the mechanism that tells you *which template to fix next*, which is more useful to you right now (early, iterating on `templates.py`/`constructs.py`) than a single pass/fail number.

## 5. Gate to FREEZE

An item is eligible for the immutable frozen set only when:
1. Layer A: `auto_pass` on all checks in §2, including vision-necessity baselines, and
2. Layer B: `status: verified` (or adjudicated to verified) with no open discrepancy, and
3. `vision_necessary == true` is recorded, satisfying invariant #4.

`allocentric_encoding` remains `unsupported` until trusted object-facing / `edges_object_frame` exists — do not promote those items to FREEZE.

## 6. Open decisions for you to set (not code-decidable)

- IAA threshold per construct (§3.4).
- Whether Layer B is 100% or sampled, and when to switch (§3.5).
- Who adjudicates disagreements (§3.3) — needs to be a fixed, named role, not "whoever's free."
- Baseline models/thresholds for the vision-necessity gate (§2.4) — which blind-LLM and captioner, and what "above chance" margin you'll accept.