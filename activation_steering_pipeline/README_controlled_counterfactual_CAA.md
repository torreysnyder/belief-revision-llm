# Controlled Counterfactual CAA Pipeline

## Purpose

This document specifies the next stage of the belief-revision experiment
after completion of the behavioral pilot with the local Qwen assistant.
The goal is to construct **two distinct Contrastive Activation Addition
(CAA) vector families** from controlled counterfactual evidence pairs:

1.  **Core-belief CAA** --- a direction associated with processing
    evidence that challenges the vignette's core belief.
2.  **Auxiliary-belief CAA** --- a direction associated with processing
    evidence that challenges an auxiliary belief.

The central design principle is to construct the contrast
experimentally. CAA examples should **not** be defined retrospectively
as conversations in which the assistant happened to revise versus
resist. Instead, each contrastive example is a matched counterfactual
pair branching from the same conversation state and differing primarily
in the evidential manipulation.

------------------------------------------------------------------------

## 1. Inputs from the behavioral pilot

Use the completed pilot JSONL as the source of conversation states and
experimental metadata. For each eligible evidence-injection point,
retain enough information to reconstruct the exact Qwen prompt
immediately before the assistant response, including:

-   conversation/cell identifier and replicate;
-   vignette and conversation history;
-   phase and turn index;
-   evidence ID, role, target, strength, and credibility where
    applicable;
-   persona/style condition;
-   affect;
-   institutional trust;
-   conversational goal;
-   belief-anchor condition;
-   auxiliary-belief mapping and active auxiliaries;
-   behavioral probe measurements for later validation.

Do not overwrite the pilot outputs. The counterfactual dataset and
activation dataset should be separate derived artifacts with traceable
links back to the original conversation IDs.

------------------------------------------------------------------------

## 2. Identify eligible evidence-injection states

Create two non-overlapping collections of source states.

### 2.1 Core-belief source states

Select evidence-injection points at which the incoming evidence targets
the **core belief**. Preserve the complete conversation history
immediately before the evidence-bearing user turn.

### 2.2 Auxiliary-belief source states

Select evidence-injection points at which the incoming evidence targets
an **auxiliary belief**. Record which canonical/display auxiliary is
targeted and whether any gating condition has been satisfied.

Do not pool Core and Auxiliary examples during vector construction.

------------------------------------------------------------------------

## 3. Generate controlled counterfactual evidence pairs

For every eligible source state, branch from the **same preceding
conversation history** and generate a matched pair.

### Experimental branch

Use the real disconfirming evidence appropriate to the target belief.

### Counterfactual control branch

Replace the disconfirming evidence with a neutral or supportive
substitute while holding nuisance variables as constant as possible.

Each pair should be matched on:

-   complete preceding dialogue context;
-   evidence target (Core or the same Auxiliary);
-   phase and turn position;
-   persona/style;
-   affect;
-   institutional trust;
-   conversational goal;
-   approximate token/word length;
-   grammatical structure and level of specificity where feasible;
-   evidence strength and credibility, except for the polarity/content
    feature intentionally manipulated.

Prefer **minimal counterfactual edits** over independently generated
control messages. The objective is for evidential polarity/relevance to
be the dominant difference between the two prompts.

### Neutral and supportive controls

Retain `control_type` explicitly rather than automatically pooling the
two control classes.

-   `disconfirming ↔ neutral` estimates a challenge/evidence-related
    contrast.
-   `disconfirming ↔ supportive` places the conditions on opposite sides
    of belief consistency and may yield a stronger but somewhat
    different direction.

Initially analyze these separately. Pool only if subsequent analyses
show that their layer-wise directions and effects are sufficiently
similar.

------------------------------------------------------------------------

## 4. Produce two contrastive datasets

The resulting data structure should contain two datasets.

### Dataset A --- Core CAA

For each source state:

`core_disconfirming ↔ core_neutral/supportive`

### Dataset B --- Auxiliary CAA

For each source state:

`aux_disconfirming ↔ aux_neutral/supportive`

Suggested pair-level fields:

``` text
pair_id
conversation_id
cell_id
replicate_index
vignette_id
phase
global_turn_index
contrast_type          # core | auxiliary
control_type           # neutral | supportive
target_belief          # core | A1 | A2 / canonical auxiliary ID
evidence_id
style_id
affect
trust_in_institutions
conversation_goal
belief_anchor_level
experimental_text
control_text
experimental_prompt
control_prompt
```

Add automatic quality checks for approximate length matching, accidental
target changes, missing metadata, duplicate pairs, and malformed chat
histories. Manually inspect a stratified sample before activation
extraction.

------------------------------------------------------------------------

## 5. Split construction and evaluation data

Prevent circular evaluation by separating examples used to estimate CAA
vectors from examples used to test steering.

At minimum create:

-   **CAA construction split** --- used to estimate layer-wise
    directions;
-   **held-out evaluation split** --- never used when estimating the
    vectors.

Split at a level that minimizes leakage between near-duplicate
counterfactuals. Where sample size permits, consider holding out
complete conversation states or experimental cells rather than randomly
separating individual rows.

Behavioral belief-change scores should be used for
**validation/evaluation**, not to define the positive and negative CAA
classes.

------------------------------------------------------------------------

## 6. Reconstruct Qwen inputs

For each branch, reconstruct the exact chat context that Qwen would
receive immediately before generating its response.

Important reproducibility controls:

-   freeze the exact Qwen model and tokenizer revision;
-   use the same chat template as the behavioral experiment;
-   preserve message roles and ordering;
-   record tokenizer/model revision and extraction code version;
-   use deterministic forward passes for activation extraction;
-   verify that the reconstructed original branch tokenizes identically
    to the corresponding behavioral prompt wherever possible.

The expensive user-simulator API does not need to be rerun if the
required counterfactual messages have already been generated and stored.

------------------------------------------------------------------------

## 7. Extract residual-stream activations with `extract_caa_activations.py`

After `generate_caa_contrastive_pairs.py` has produced and quality-checked the
Core and Auxiliary pair datasets, the next computational stage is activation
extraction. Implement this stage in a separate script,
`extract_caa_activations.py`, so that activation collection remains independent
from subsequent choices about vector construction, filtering, normalization,
layer selection, and steering.

### 7.1 Inputs

The primary inputs are:

``` text
data/caa_pairs_core.jsonl
data/caa_pairs_auxiliary.jsonl
data/caa_split_manifest.json
```

The pair JSONLs are the authoritative textual records. They contain the
experimental and matched control prompts plus the experimental metadata needed
to interpret each pair. The split manifest assigns each conversation to a
frozen `train`, `validation`, or `test` split.

Only pairs that pass hard QC should be used for the primary extraction dataset.
Do **not** automatically discard all soft-flagged pairs. Preserve their soft
flags as metadata so sensitivity analyses can later exclude particular warning
classes without rerunning Qwen.

### 7.2 Freeze the data split before extraction

The original pilot conversations are not intrinsically labeled as train,
validation, or test. Create the split once, save it, and reuse it throughout
the CAA analysis.

Assign splits by **conversation ID**, not by individual pair. Every Core,
Auxiliary, neutral, supportive, and phase-specific pair derived from the same
conversation must inherit the same split. This prevents closely related
counterfactuals from leaking between CAA construction and evaluation.

A reasonable initial allocation is approximately 70% train, 15% validation,
and 15% test, with approximate stratification over important crossed
experimental conditions where feasible.

Save the assignment in `data/caa_split_manifest.json`. At minimum it should
contain a mapping from `conversation_id` to `split`, together with dataset-level
provenance such as:

``` text
split_seed
split_method
train_fraction
validation_fraction
test_fraction
stratification_variables
creation_timestamp
```

Use the splits as follows:

- **train** --- construct the CAA vectors;
- **validation** --- choose development decisions such as candidate layer,
  steering coefficient, and normalization;
- **test** --- final held-out causal steering evaluation; never use it to tune
  the vector definition, layer, or steering strength.

### 7.3 Reconstruct and verify Qwen inputs

For every QC-passing pair, reconstruct both branches using the exact frozen
Qwen model, tokenizer, and chat template:

``` text
experimental_prompt -> Qwen
control_prompt      -> Qwen
```

The two branches should differ only in the controlled counterfactual
manipulation encoded by the pair dataset. Preserve message roles and ordering,
and verify the reconstructed/rendered prompts against the stored pair records
where possible.

Activation extraction is a deterministic forward-pass operation. The remote
user simulator and judge do not need to be rerun.

### 7.4 Initial activation definition

For the initial CAA experiment, extract the residual-stream representation at
the **final prompt token immediately before assistant generation** for every
transformer layer.

For pair `i` and layer `l`, save:

``` text
h_experimental[i,l]
h_control[i,l]
```

with tensor shape:

``` text
[n_layers, hidden_dim]
```

Also save the within-pair difference:

``` text
pair_difference[i,l] = h_experimental[i,l] - h_control[i,l]
```

Do not save only the difference tensor. Keeping both original activations
allows later changes to normalization, diagnostics, exclusions, and vector
construction without repeating extraction.

Keep the extraction code flexible enough to support later alternatives such
as mean pooling over evidence tokens, the final evidence token,
assistant-response token positions, or selected semantic spans. The first
analysis should nevertheless use one prespecified activation definition to
avoid multiplying researcher degrees of freedom.

### 7.5 Information saved for every activation pair

Each activation record should contain the following pair-level information.
Fields unavailable in a particular pair record should be represented
explicitly as missing rather than silently inferred.

#### Identity and provenance

``` text
pair_id
source_state_id
conversation_id
cell_id
replicate_index
```

#### Frozen split

``` text
split                       # train | validation | test
```

#### Contrast definition

``` text
contrast_type               # core | auxiliary
control_type                # neutral | supportive
phase
turn_in_phase
global_turn_index
```

#### Evidence and belief target

``` text
evidence_id
evidence_role
evidence_strength
evidence_credibility
target_belief
target_belief_canonical     # core | A1 | A2, as applicable
```

#### Experimental factors

Preserve relevant crossed factors when they are available in the pair dataset:

``` text
style_id
initial_affect
trust_in_institutions
conversation_goal
belief_anchor_level
aux_order_condition
```

#### Quality-control metadata

``` text
qc_pass
qc_hard_flags
qc_soft_flags
```

The primary extraction set should normally contain `qc_pass == true` records,
but retaining the QC fields makes the activation artifact self-describing and
supports later sensitivity analyses.

#### Tokenization and extraction position

``` text
experimental_seq_len
control_seq_len
experimental_token_id
control_token_id
token_position              # initially: final_prompt_token
```

The token IDs record the actual token at the extraction position for each
branch. Sequence lengths are useful for detecting unexpected prompt-rendering
or tokenization differences.

#### Activation tensors

``` text
experimental_activation     # [n_layers, hidden_dim]
control_activation          # [n_layers, hidden_dim]
pair_difference             # experimental - control
```

#### Model and extraction provenance

These fields may be stored per record when convenient, but invariant values
should preferably live once in `activation_manifest.json` to avoid redundant
storage:

``` text
model_name
model_revision
tokenizer_name
tokenizer_revision
activation_location
token_position
n_layers
hidden_dim
activation_dtype
extraction_script_version
extraction_timestamp
chat_template_hash
prompt_match_verified
```

`activation_location` must state precisely what representation was captured
(e.g. transformer-block output/layer-boundary hidden state), because “residual
stream activation” is otherwise ambiguous.

### 7.6 Do not duplicate the full prompts in the tensor files

The complete experimental/control prompts and evidence text should remain in
`caa_pairs_core.jsonl` and `caa_pairs_auxiliary.jsonl`. The activation files
should link back to those authoritative records through `pair_id` and
`source_state_id` rather than duplicating large strings inside tensor files.

### 7.7 Activation outputs

Write separate Core and Auxiliary activation artifacts:

``` text
activations/core_activations.pt
activations/auxiliary_activations.pt
activations/activation_manifest.json
```

The `.pt` files contain the pair-level metadata and activation tensors. The
manifest contains dataset-wide extraction metadata and counts.

Recommended `activation_manifest.json` fields include:

``` text
model_name
model_revision
tokenizer_name
tokenizer_revision
activation_location
token_position
layers_extracted
n_layers
hidden_dim
activation_dtype
chat_template_hash
extraction_script_version
extraction_timestamp
core_input_file
auxiliary_input_file
split_manifest_file
n_core_pairs_seen
n_core_pairs_extracted
n_auxiliary_pairs_seen
n_auxiliary_pairs_extracted
n_qc_excluded
device
torch_version
transformers_version
```

Record hashes for the input pair files, split manifest, model revision, and
extraction script where practical. This makes each activation artifact
traceable to the exact inputs and code that produced it.

### 7.8 Keep extraction separate from CAA vector construction

`extract_caa_activations.py` should **not** average examples into the final CAA
vectors. Its job is to create a reusable, frozen activation dataset.

A subsequent `build_caa_vectors.py` stage should consume the activation files
and construct vectors using only the permitted split. This separation allows
changes to QC exclusions, neutral-versus-supportive treatment, normalization,
resampling, and vector construction without rerunning Qwen.

Initially preserve four analytically distinct contrasts:

``` text
Core:      disconfirming - neutral
Core:      disconfirming - supportive
Auxiliary: disconfirming - neutral
Auxiliary: disconfirming - supportive
```

Pool neutral and supportive controls only if later vector diagnostics and
behavioral results justify doing so.

------------------------------------------------------------------------

## 8. Construct paired CAA directions

Calculate the within-pair activation difference first:

``` text
d[i,l] = h_real[i,l] - h_control[i,l]
```

Then average the paired differences for each layer.

### Core vector

``` text
v_core[l] = mean_i(d_core[i,l])
```

### Auxiliary vector

``` text
v_aux[l] = mean_i(d_aux[i,l])
```

This produces two layer-wise vector families:

``` text
V_core = {v_core[0], ..., v_core[L]}
V_aux  = {v_aux[0],  ..., v_aux[L]}
```

Store vectors together with the dataset version, model revision, layer
index, activation location, control type, number of pairs, and any
normalization applied.

------------------------------------------------------------------------

## 9. Characterize the vectors before intervention

Before steering generation, perform descriptive checks.

Recommended analyses include:

-   vector norm by layer;
-   cosine similarity of Core vectors across resampled subsets;
-   cosine similarity of Auxiliary vectors across resampled subsets;
-   Core--Auxiliary cosine similarity by layer;
-   neutral-control versus supportive-control vector similarity;
-   bootstrap confidence/stability estimates;
-   simple held-out classification or projection tests showing whether
    the direction separates the paired conditions.

These analyses help identify candidate layers and determine whether Core
and Auxiliary contrasts appear representationally distinct.

Do not select the final steering layer solely because it maximizes the
desired behavioral outcome on the evaluation set. Layer selection should
be prespecified or based on construction/validation data.

------------------------------------------------------------------------

## 10. CAA steering intervention

At generation time, intervene on the corresponding residual stream using
a layer-wise CAA vector:

``` text
h'[l] = h[l] + alpha * v[l]
```

Test both signs:

``` text
+CAA: h'[l] = h[l] + alpha * v[l]
-CAA: h'[l] = h[l] - alpha * v[l]
```

Use several steering coefficients (`alpha`) including zero as the
baseline. Begin with a small layer/alpha sweep on development examples
before committing to the full held-out evaluation.

Monitor not only belief-related outcomes but also generation quality,
repetition, incoherence, response length, and other nonspecific
behavioral changes.

------------------------------------------------------------------------

## 11. Primary causal tests

The two-vector design enables four particularly informative
interventions.

  -----------------------------------------------------------------------
  Steering vector         Evaluation context      Main question
  ----------------------- ----------------------- -----------------------
  Core → Core             Core-belief challenge   Does Core steering
                                                  causally alter
                                                  core-belief revision?

  Auxiliary → Auxiliary   Auxiliary-belief        Does Auxiliary steering
                          challenge               causally alter
                                                  auxiliary-belief
                                                  revision?

  Core → Auxiliary        Auxiliary-belief        Does the Core direction
                          challenge               generalize to auxiliary
                                                  revision?

  Auxiliary → Core        Core-belief challenge   Does the Auxiliary
                                                  direction affect core
                                                  revision?
  -----------------------------------------------------------------------

The cross-steering conditions are important for determining whether the
representations are specific to Core versus Auxiliary belief processing
or instead reflect a more general evidence/revision direction.

------------------------------------------------------------------------

## 12. Behavioral evaluation

Evaluate steered and unsteered generations using the same
belief-revision outcomes wherever possible.

Primary outcomes can include:

-   change in core-belief score;
-   change in targeted auxiliary-belief score;
-   `changed_node`;
-   assistant stance;
-   epistemic confidence;
-   specificity of the effect to the targeted belief;
-   response-quality controls.

Compare at least:

``` text
-CAA  vs  baseline  vs  +CAA
```

The strongest evidence for a causal mechanism would be a dose-sensitive,
directionally coherent change in the theoretically predicted belief
outcome without a comparable degradation in general response quality.

------------------------------------------------------------------------

## 13. Controls and robustness checks

Include controls that distinguish a meaningful CAA effect from generic
activation perturbation.

Candidate controls:

-   random direction matched for vector norm;
-   shuffled pair labels;
-   vectors constructed from irrelevant phases/targets;
-   alternative layers;
-   positive versus negative steering;
-   neutral-derived versus supportive-derived CAA vectors;
-   held-out experimental cells;
-   Core/Auxiliary cross-steering;
-   multiple random seeds for generation.

If feasible, repeat the strongest result on an additional vignette/model
as a generalization test.

------------------------------------------------------------------------

## 14. Relationship to subsequent SAE work

CAA should precede the more expensive SAE analysis.

The CAA stage establishes whether a relatively simple residual-stream
direction:

1.  reliably distinguishes the controlled counterfactual conditions;
2.  predicts relevant behavioral differences on held-out examples; and
3.  causally changes belief-revision behavior when intervened upon.

The SAE stage can then ask whether the CAA direction decomposes into
interpretable sparse features, whether particular SAE features
distinguish Core from Auxiliary processing, and whether manipulating
those features reproduces or refines the CAA intervention.

Avoid using the SAE simply to rediscover a contrast that has not first
been shown to have a robust causal behavioral effect.

------------------------------------------------------------------------

## 15. Recommended implementation order

1.  **Freeze the completed pilot dataset and model/tokenizer versions.**
2.  **Identify Core and Auxiliary evidence-injection source states.**
3.  **Generate matched neutral/supportive counterfactual messages.**
4.  **Run automated and manual pair-quality checks.**
5.  **Create and freeze conversation-level train/validation/test splits.**
6.  **Reconstruct and verify Qwen prompts for both branches.**
7.  **Run `extract_caa_activations.py` and save pair-level activations plus an activation manifest.**
8.  **Run `build_caa_vectors.py` on the construction split to compute paired Core and Auxiliary CAA vectors.**
9.  **Measure vector stability and Core--Auxiliary similarity.**
10. **Select candidate intervention layers without using the held-out
    outcome data.**
11. **Run ±CAA and alpha sweeps on development examples.**
12. **Run preregistered Core→Core and Auxiliary→Auxiliary held-out
    tests.**
13. **Run Core→Auxiliary and Auxiliary→Core cross-steering tests.**
14. **Run random/shuffled-direction and generation-quality controls.**
15. **Only then proceed to SAE training and feature-level
    intervention.**

------------------------------------------------------------------------

## Expected artifacts

A clean implementation should produce separate, versioned artifacts
rather than modifying the behavioral pilot files:

``` text
data/
  pilot_original.jsonl
  caa_pairs_core.jsonl
  caa_pairs_auxiliary.jsonl
  caa_pairs_qc_report.json
  caa_pairs_manual_review.csv
  caa_pairs_manifest.json
  caa_split_manifest.json

activations/
  core_activations.pt
  auxiliary_activations.pt
  activation_manifest.json

vectors/
  core_caa_vectors.pt
  auxiliary_caa_vectors.pt
  vector_metadata.json

results/
  vector_diagnostics.csv
  steering_development.csv
  steering_heldout.csv
  steering_controls.csv
```

Record hashes/version identifiers where practical so every steering
result can be traced to the exact counterfactual dataset and vector
construction procedure.

------------------------------------------------------------------------

## Key design rule

> **Construct the Core and Auxiliary CAA directions from controlled,
> paired counterfactual evidence manipulations; use observed belief
> revision as an outcome for validation and causal testing, not as the
> criterion that defines the contrastive classes.**
