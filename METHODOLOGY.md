# HeliosTune methodology v1

## Status and normative language

This is HeliosTune's active, partially implemented evidence-control design target. The document is normative only for artifacts whose exact schema literal is `heliostune.protocol/1`, `heliostune.bundle/1`, `heliostune.analyzer-manifest/1`, or `heliostune.verification-record/1`. It does not upgrade older artifacts or claim that the current CLI or every study implements the contract; see [Implementation status](#11-implementation-status). Frozen protocols, bundles, records, and published evidence remain immutable, while new implementation work may land at new versioned paths without rewriting those bytes or claims.

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are to be interpreted as described by RFC 2119. Each mandatory rule below states both the failure it prevents and the machine check that enforces it. A verifier has no warning-only path for a failed mandatory rule.

## 1. Purpose, scope, and non-goals

HeliosTune is an evidence control plane between candidate generation and a published performance claim:

```text
Study plugin → resolved protocol → immutable freeze → executor
             → sealed EvidenceBundle → VerificationRecord
             → registered analyzer replay → typed claim → publication
```

It controls study identity, semantic closure, role separation, execution accounting, evidence custody, claim eligibility, and reproducible analysis. A plugin supplies domain choices such as workloads, candidate space, comparator applicability, numerical tolerances, timing strata, practical margins, estimators, and sampling counts.

Methodology v1 does not:

- define a universally correct corpus size, error tolerance, cache regime, practical margin, cluster count, or statistical estimator;
- replace Triton, CUTLASS, framework, vendor-library, task-suite, timer, or cloud-scheduler implementations;
- turn an exhaustive search of one enumerated space into a hardware ceiling or a universal kernel claim;
- turn a fixed workload census into a population sample, repeated launches into independent workloads, or one GPU instance into a GPU population;
- make data comparable when their semantic, numerical, timing, or execution contracts differ;
- authenticate a provider that supplies no verifiable receipt, or promise that retained measurements can be recollected without equivalent hardware; or
- retroactively repair missing fields, provenance, controls, or eligibility in a legacy artifact.

### Evidence classes

| Class | Permitted use | Completeness and separation | Claim eligibility |
|---|---|---|---|
| `exploratory` | Debugging, protocol refinement, descriptive screening, including a five-minute child protocol. | May use a declared prefix and may end incomplete. Per-cell semantics and attempt accounting are unchanged. | `descriptive` only. It is permanently non-promotable. |
| `engineering_gate` | A frozen go/stop decision, cost screen, correctness gate, or implementation screen. | The declared gate is complete. Same-bank selection and scoring are allowed only when disclosed. | Descriptive output and `stopped`; never an inferential win from same-bank evidence. |
| `confirmatory` | A predeclared claim on untouched evaluation evidence. | All planned cells close; estimator, margin, multiplicity, stopping, roles, and eligible provenance were frozen before dispatch. | Any claim kind whose additional rules below pass. |

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| EC1 | A protocol **MUST** declare exactly one evidence class, and a bundle **MUST NOT** claim a stronger class than its protocol. | Relabeling an exploratory or stopped screen after seeing favorable results. | Exact enum validation and protocol-digest equality; the claim validator compares every claim with the frozen class. |
| EC2 | An `exploratory` protocol **MUST** emit only `descriptive` claims and **MUST NOT** be promoted, merged, or reused as confirmatory input. | Exploration masquerading as preregistration and outcome-selected reuse. | Parent/child and input-digest graph traversal rejects exploratory observations in an inferential claim and rejects class-changing revisions. |
| EC3 | An `engineering_gate` using the same bank for selection and scoring **MUST** label the result selection-optimistic and **MUST NOT** emit an inferential decision. | The optimizer's curse being reported as an independent win. | Bank-role and exposure-ledger check restricts outputs to `descriptive` or `stopped` and requires the limitation code `same_bank_selection_optimism`. |
| EC4 | A `confirmatory` claim **MUST** use untouched evaluation evidence and frozen estimator, margin, multiplicity, stopping, and provenance requirements. | Post-hoc inference, optional stopping, and target leakage. | Dispatch-time exposure ledger plus byte-digest comparison of analysis and split objects; claim eligibility requires `VERIFIED` bundle closure. |

## 2. Identity, strict data, lifecycle, and revisions

### 2.1 Strict JSON and closed references

Both v1 schemas use strict JSON. Objects have exact field sets for their schema version; unknown or duplicate keys are rejected. Booleans are not integers. Integers and numbers have exact validators; all numbers are finite. Digests are lowercase 64-character hexadecimal SHA-256 strings. Timestamps are RFC 3339 strings. Paths are normalized, unique, relative, and non-escaping. A digest reference is valid only when the referenced bytes are present in the protocol package or bundle and recompute to that digest.

`heliostune.protocol/1` is the non-retroactive protocol schema. Its required semantic identity includes:

```text
schema, study_id, revision, created_at, evidence_class,
parent_protocol_sha256, plugin, semantic, analysis, execution
```

The exact v1 field map is:

```text
schema: literal "heliostune.protocol/1"
study_id: nonblank string
revision: integer >= 1
created_at: RFC 3339 string
evidence_class: exploratory | engineering_gate | confirmatory
parent_protocol_sha256: digest | null
plugin: {id: string, version: string, artifact_sha256: digest}
semantic: {
  workloads_sha256: digest, candidates_sha256: digest,
  comparators_sha256: digest, splits_sha256: digest,
  numerics_sha256: digest, timing_sha256: digest
}
analysis: {analyzer_sha256: digest, claims: ClaimSpec[]}
execution: {
  executor_api: string, expected_cells_sha256: digest,
  expected_cell_count: integer >= 0,
  environment_predicate_sha256: digest, failure_policy_sha256: digest,
  retry_policy: none | pre_measurement_infrastructure,
  max_physical_attempts: integer >= 1, wall_limit_s: integer >= 1,
  paid_plan_sha256: digest | null
}

ClaimSpec: {
  claim_id: string,
  kind: descriptive | superiority | inferiority | noninferiority |
        equivalence | scoped_exhaustive_dominance | transfer_benefit,
  candidate_id: string, comparator_id: string, reference_id: string | null,
  estimand_ast_sha256: digest, units: string, direction: higher | lower,
  scope_sha256: digest, population_sha256: digest,
  delta: finite number >= 0 | null, alpha: finite number in (0, 0.5) | null,
  multiplicity_family: string | null,
  stopping: none | fixed_n | confidence_sequence
}
```

For `retry_policy: none`, `max_physical_attempts` is exactly one. A pre-measurement-infrastructure retry policy allows at least two attempts but remains subject to the stricter execution rules in §6. Descriptive claims use null `delta`, `alpha`, and `multiplicity_family` with `stopping: none`; inferential ClaimSpecs provide all four fields, and noninferiority/equivalence use a strictly positive `delta`. Candidate, comparator, and non-null reference IDs identify distinct roles. A scoped exhaustive-dominance claim has a non-null reference.

`semantic` binds the workload, candidate, comparator, split, numerics, and timing objects. `analysis` binds the analyzer and claim specifications. `execution` binds the executor API, expected-cell plan, environment predicate, failure and retry policies, wall limit, and optional paid plan. The schema implementation is the authoritative exact field/type definition; this document defines their semantics.

### 2.2 Lifecycle

The only forward lifecycle is:

```text
DRAFT → RESOLVED → FROZEN → DISPATCHED → SEALED → VERIFIED → ANALYZED → PUBLISHED
```

- `DRAFT`: editable author input; no stable evidence identity.
- `RESOLVED`: all plugin defaults, generated manifests, capability decisions, and referenced bytes are explicit; no target exposure has occurred.
- `FROZEN`: canonical bytes and a digest identify an immutable revision.
- `DISPATCHED`: target exposure has begun, including an invocation that fails before returning a row.
- `SEALED`: no more attempts may be appended; every attempt is terminal or explicitly orphaned/unresolved.
- `VERIFIED`: schema, closure, custody, semantic, eligibility, and replay checks completed.
- `ANALYZED`: the frozen analyzer produced typed claims from the verified bundle.
- `PUBLISHED`: a verified content-addressed root and catalog generation are visible atomically.

`outcome = completed | failed | aborted` is orthogonal to lifecycle. Failed and aborted work still advances through sealing and may publish failure evidence. `ARCHIVED`, `withdrawn`, and retention status are catalog metadata, not lifecycle states and not evidence erasure.

A change after resolution creates a higher integer `revision`. A change after freeze creates a new protocol digest and an explicit `supersedes` relation; it never alters or replaces the old bytes. Revisions do not combine evidence unless a later frozen protocol explicitly declares a valid synthesis design.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| ID1 | A v1 document **MUST** use its exact schema literal, exact JSON types, closed field set, finite numbers, lowercase SHA-256 form, and closed references. | Ambiguous parsing, schema smuggling, digest substitution, and dangling evidence. | `heliostune.validation` exact-type/field validators, duplicate-key detection, digest regex, path normalization, and recursive reference resolution. |
| LC1 | Lifecycle events **MUST** follow the eight-state graph without a backward edge, and `outcome` **MUST** remain orthogonal. | Reopening a run, erasing failure, or treating publication as proof of success. | Transition-table validator checks monotonic sequence numbers, legal predecessor, state, outcome, timestamp, actor, and protocol digest. |
| LC2 | Freeze bytes **MUST** remain immutable after `FROZEN`, and every later record **MUST** carry the same protocol digest. | Outcome-driven protocol edits and mixed-revision evidence. | Canonical-byte SHA-256 recomputation at dispatch, seal, verification, analysis, and publication. |
| LC3 | Any semantic change after freeze **MUST** create a new study revision with a `supersedes` edge and **MUST NOT** hide the older revision. | Silent reruns and historical replacement. | Registry uniqueness and monotonic-revision checks; catalog expected-set equality includes both revisions and all outcomes. |
| LC4 | Dispatch **MUST** mark the start of target exposure even when execution fails before measurement. | Treating a failed paid invocation as a cost-free pilot. | Intent journal timestamp precedes provider action; exposure ledger rejects a later freeze under the same revision. |
| LC5 | Failed or aborted work **MUST** seal its partial rows, attempts, costs, and reason rather than disappear. | Survivor bias and hidden operational failure. | Closure compares journal, provider calls, terminal/orphan states, ledger, and artifact inventory before allowing a terminal outcome. |

## 3. Semantic protocol

Resolution converts plugin input into explicit content-addressed objects. Local and remote executors consume the same semantic objects. Transport, provider locator, hardware predicate, wall/cost cap, and executor identity are explicit execution differences; they do not reinterpret workloads, arms, numerics, timing, analysis, or failure contribution.

### 3.1 Plugin and interface contracts

A study plugin exposes stable, versioned behavior equivalent to:

```text
StudyPlugin.resolve(draft) -> ResolvedProtocol
StudyPlugin.compile(protocol) -> ExecutionEnvelope
WorkloadProvider.materialize(case_id, seed) -> Inputs
Arm.prepare(case, environment) -> PreparedArm
Arm.execute(prepared) -> Outputs
Executor.preflight(envelope) -> PreflightRecord
Executor.dispatch(envelope) -> AttemptRecords
Executor.reconcile(idempotency_key) -> AttemptState
Executor.cancel(idempotency_key) -> AttemptState
Analyzer.analyze(protocol, sealed_bundle) -> ClaimSet
BundleVerifier.verify(root, registry, policy) -> VerificationRecord
```

Candidate and comparator arms share the same prepare/execute/result boundary. Executors transport an envelope and report actual state; they do not select candidates, change tolerances, shorten warmups, drop failed cells, or render claims. An analyzer is a pure function of a frozen protocol and sealed bundle.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| PL1 | A plugin **MUST** bind its stable ID, version, artifact digest, API version, and every generated semantic object. | Plugin substitution and unrepeatable implicit defaults. | Resolver repeats under an isolated process; protocol validation compares byte-identical output and all referenced digests. |
| PL2 | Candidate and comparator adapters **MUST** implement the same declared input, prepare, execute, output, numerics, and timing boundaries. | Measuring hidden setup for one arm but not another. | Compiled call-plan comparison requires one shared boundary contract ID for every arm in a ratio or contrast. |
| PL3 | An executor **MUST** return the compiled semantic envelope digest without reinterpretation. | Local/remote semantic drift. | Independent local and remote compilation fixture compares semantic call-plan hashes; returned rows bind the envelope hash. |
| PL4 | An analyzer **MUST** be deterministic and pure over the frozen protocol plus sealed bundle. | Outcome-dependent external inputs and irreproducible headlines. | Network-disabled replay with fixed locale/time zone regenerates ClaimSet and report bytes and compares digests. |

#### 3.1.1 Plugin and suite declarations

`heliostune.plugin/1` and `heliostune.suite/1` are strict, additive declaration
schemas. They close an initial operation vocabulary and make plugin identity,
suite custody, case semantics, arm applicability, numeric contracts, reference
arms, and expected-cell order inspectable without importing or executing
plugin code. The narrow reference templates do not encode or claim a complete
performance-baseline hierarchy; that remains a promotion and evidence
requirement under §3.4. The schemas do not alter the exact
`heliostune.protocol/1` field map. A protocol can bind the exact plugin root
through `plugin.artifact_sha256`; the plugin root, in turn, binds normalized
relative suite paths and their exact SHA-256 values.

Standalone plugin validation resolves plugin → suite references from the
filesystem. Generic bundle verification now traverses the same edge exclusively
through the additive inventoried bytes described in §7.1. Only a complete,
descriptor-selected inventory earns `plugin_suite_custody: checked`; bundles
without the reserved roles remain explicitly `not_checked`.

Declaration status is multi-axis:

| Axis | Meaning |
|---|---|
| Vocabulary | A closed domain/dtype/case token may appear in a declaration. |
| Schema | Exact fields, types, cross-rules, references, and hashes validate. |
| Template | A named suite has frozen cases, arms, contracts, and expected cells. |
| Bundle custody | Opted-in inventoried plugin and suite bytes close their ordered identities and digests internally; this does not establish authorship or execution. |
| Backend capability | Local and remote probes separately record `unprobed`, `available`, or `unavailable`. |
| Correctness observation | Retained output evidence passed the frozen contract for an exact cell key. |
| Performance observation | Retained timing evidence was collected after that passing gate. |

The axes are not implications. Capability states `available` and `unavailable`
require an evidence SHA-256; `unprobed` requires null evidence. Availability
does not establish correctness. Correctness does not establish performance.
Schema or template validation establishes neither.

The closed domain vocabulary may name `dense_gemm`, `fused_mlp`,
`rmsnorm_residual`, `attention`, `kv_cache`, `moe`, and `quantized_linear`.
The closed dtype vocabulary may name `fp32`, `tf32`, `fp16`, `bf16`,
`fp8_e4m3fn`, `fp8_e5m2`, `int8`, `int4`, and `uint4`. Vocabulary membership
is not backend support.

Only `gated_mlp_epilogue.v1` and `residual_rmsnorm.v1` are frozen initial
executable-suite templates. Their numeric contracts permit FP16/BF16
input/storage, FP32 accumulation, FP16/BF16/FP32 stored output, null
quantization, and disabled TF32. The schema can represent advanced dtype names,
exact four-bit packing order/axis, scale dtype/layout/granularity, and
calibration metadata, but those dtypes cannot appear in either initial
template. Missing zero-point, dequantization, rounding/saturation,
instruction-readback, reference, and error-policy choices must be closed in a
separate reviewed schema/suite revision before promotion.
The committed declarations are explicitly labeled
`reference_template_not_execution_freeze`; their template identity and hashes
are frozen, but no runtime capability or dispatch authorization follows.

Gated-MLP cases close activation (`silu | gelu`), gate/up layout
(`separate | packed`), bias and residual booleans, output arity one, and the
ordered fusion boundary. Residual-RMSNorm cases close finite positive epsilon,
gamma presence, residual position (`pre | post`), output arity one or two, and
the ordered fusion boundary. Arm shape constraints are inline triples of
dimension, operation (`divisible_by | min | max | equal`), and integer value.

The expected-cell list is a plan rather than an observation. It must place a
correctness-stage cell before every timing-stage cell having the same case,
arm, and input seed. An executor must additionally require a retained passing
correctness observation for that exact key before dispatching timing. It may
not treat static order, capability availability, or a pass on another seed as
the runtime gate.

The complete declaration fields, baseline/promotion requirements, staged
attention/KV-cache, quantized-linear, MoE and FP8 candidates, template
identities, hashes, implementation order, and inspection commands are in
[Experiment scope](EXPERIMENT_SCOPE.md).

### 3.2 Workloads

The workload object defines whether cases form a fixed census or a sample from a named population. Each case records stable ID; source model/node or generator provenance; semantic graph/DAG hash; shapes; dtype, layout, strides, and alignment; dynamic-state bin; input generator and seed contract; weight; family and leakage-component IDs; and expected outcome.

Weights are fixed without candidate outcomes. Equal weights are the default when no attested production distribution exists. A production-utility claim reports both equal-family macro aggregation and a frozen production-time-weighted gate. Coverage and failures remain in the denominator according to the frozen estimand; a failed case is never silently removed.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| W1 | The workload manifest **MUST** be immutable, content-addressed, split-aware, and weighted before target exposure. | Toy-shape selection, outcome-dependent weighting, and corpus drift. | Recompute manifest digest, weight sum and case identities; exposure timestamps precede no manifest or weight mutation. |
| W2 | Every case **MUST** declare semantic, shape, dtype/layout/stride, provenance, generator/seed, family, leakage, and expected-outcome fields. | Accidental pooling of different operations and undetected duplicates. | Exact case schema plus uniqueness and semantic-hash consistency checks. |
| W3 | Coverage and failure contribution **MUST** be frozen as part of the estimand, and failed cases **MUST NOT** be dropped post hoc. | Favorable complete-case bias. | Expected-case set equality and analyzer recomputation include each terminal failure according to the frozen rule. |
| W4 | A population claim **MUST** identify an actual sampling frame and inferential unit; a fixed census **MUST NOT** be relabeled as a future workload population. | Unsupported generalization beyond measured cases. | Population object, sampled cluster IDs, and resampling levels are cross-checked; fixed-census protocols can emit only census-scoped language. |

### 3.3 Candidates

A candidate object binds stable arm/config IDs, implementation and distribution digests, entry point, complete parameter schema and enumerated space, compile/setup/execute boundaries, capabilities/applicability, workspace and resource policy, precision modes, and tuning allowance. The distinction among generated candidates, selected candidates, deployable candidates, and an evaluation-only oracle is explicit.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| C1 | The candidate space and deterministic enumeration/tie-break **MUST** be frozen before evaluation. | Search-space enlargement, pruning, or tie-breaking after seeing evaluation results. | Candidate-manifest digest and evaluation exposure ledger; selected IDs must belong to the frozen manifest. |
| C2 | Each candidate **MUST** declare applicability, setup/compile/execute boundaries, workspace, precision, and tuning budget. | Hidden work, unsupported shapes, and unequal optimization effort. | Capability probe and plan compiler validate resource/tuning fields and boundary contract equality. |
| C3 | Candidate failures **MUST** produce terminal typed records and **MUST NOT** be converted to missing rows or favorable sentinel performance. | Selective omission of slow, incorrect, OOM, or compile-failing candidates. | Expected-cell closure and failure-stage enum; analyzer applies the frozen failure contribution. |

### 3.4 Comparators, applicability, and tuning parity

A comparator can have one or more roles: `correctness_oracle`, `performance_comparator`, `selection_reference`, `incumbent`, or `evaluation_oracle`. A plugin enumerates the applicable hierarchy for its operation:

1. actual production incumbent;
2. direct vendor library and applicable vendor templates;
3. compiled framework, in both declared default and tuned modes where available;
4. eager framework;
5. version-matched domain autotuner or library implementation;
6. selection-bank deployable exhaustive reference; and
7. evaluation-bank optimistic oracle, reported as diagnostic context rather than a deployable primary comparator.

This is an applicability hierarchy, not a claim that every slot exists for every operation. Structural inapplicability is decided by a capability probe that cannot inspect evaluation performance. For a tuned-vs-tuned claim, budgets include candidate count, compile time, search observations, workspace, and any offline/source amortization under the frozen cost estimand. A comparison against a production default is valid when that default is itself the declared estimand; it is not described as tuning parity.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| B1 | A confirmatory protocol **MUST** enumerate the comparator hierarchy, assign roles, and record either an applicable arm or a pre-evaluation inapplicability result for each plugin-relevant slot. | Weak-baseline shopping and silent omission of a strong known comparator. | Comparator manifest schema, capability-probe digest/timestamp, and plugin-declared hierarchy set equality. |
| B2 | Comparator applicability **MUST** be determined without evaluation outcomes, and an inapplicable arm **MUST** remain a terminal manifest record. | Calling a losing baseline inapplicable after measurement. | Capability input digest excludes evaluation rows; manifest and terminal records are frozen before dispatch. |
| B3 | Arms in a performance contrast **MUST** share semantic, numerics, timing, input-byte, and environment strata. | Ratios between different work or measurement regimes. | Cross-arm contract-ID equality and input digest checks reject the contrast. |
| B4 | A tuning-parity claim **MUST** freeze and reconcile equal or explicitly cost-normalized search, compile, observation, and workspace allowances. | Tuned-candidate versus untuned-baseline gaming. | Per-arm allowance ledger and chosen-plan hash; analyzer rejects over-budget or unreconciled arms. |
| B5 | An evaluation oracle **MUST** remain evaluation-only and **MUST NOT** be presented as a deployable selected comparator. | Conflating hindsight best with an attainable system. | Role validator rejects oracle IDs in deployable-selection or primary-comparator fields. |

### 3.5 Splits, evidence roles, and transfer leakage

Evidence roles are causal, not merely names. Tuning-visible selection evidence may choose candidate behavior. Reference evidence may choose a deployable exhaustive reference. Evaluation evidence scores the frozen candidate, reference, and comparator. Two physical banks suffice when a comparator was externally frozen without empirical target choice; otherwise distinct evidence is needed for every empirical role.

The split graph joins cases that could leak information. A plugin declares equivalence and near-match rules over lineage, task, model/fusion ancestry, semantic/config identity, exact and near shape, source capture, architecture/SKU/instance, compiler, acquisition session, and normalization/feature-fitting group. Connected components, not individual rows, are assigned to roles. The exposure ledger includes every row used to fit normalization, ranks, features, priors, hyperparameters, thresholds, comparator choice, or stopping rules.

A `transfer_benefit` design additionally isolates target-domain evaluation and uses the paired 2×2 contrast `transfer on/off × legacy-only/source-expanded candidate space`. It separates the same-space benefit of transfer from the benefit of a larger candidate space.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| S1 | Selection, empirical reference choice, and evaluation **MUST** use causally disjoint evidence components unless the result is restricted to same-bank exploratory or engineering-gate output. | Target leakage and optimizer's curse. | Split-graph connected-component intersection and exposure-ledger checks; confirmatory claims reject any shared component. |
| S2 | Evaluation evidence **MUST** remain untouched by normalization, feature fitting, ranking, hyperparameter choice, comparator choice, margin choice, and stopping design. | Indirect leakage disguised as preprocessing. | Fit-row hashes and all design-input digests are compared with evaluation row/component IDs. |
| S3 | Split equivalence and near-match rules **MUST** be frozen before role assignment. | Redefining leakage boundaries to improve results. | Split-rule digest and deterministic component regeneration from the workload manifest. |
| S4 | A transfer-benefit claim **MUST** use the paired 2×2 design, a positive frozen same-space contrast, and target-domain grouped uncertainty. | Crediting transfer for candidate-space expansion or source leakage. | Design-cell closure, interaction/contrast AST validation, and resampling-group equality. |

### 3.6 Numerical fairness

The numerical contract covers input/storage, accumulator, intermediate and output formats; precision flags and their runtime readback; arithmetic and semantic references; elementwise error envelope; nonfinite, overflow, underflow/denormal, saturation, and stochastic-output policy; adversarial and representative input distributions; input seeds; mutation/alias/stride behavior; and determinism or stochastic schedule.

At minimum, candidate and comparator are validated on identical input bytes before their timing rows become eligible. The arithmetic reference uses a declared higher-precision computation where applicable. The semantic reference independently checks that the operation, layouts, epilogue, masking, and output contract are the intended work. Aggregate error statistics supplement but cannot override an elementwise contract failure. Tolerances and case counts are plugin choices frozen before target exposure.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| N1 | Every timed candidate and comparator **MUST** pass the same frozen numerical contract on identical input bytes before timing eligibility. | Faster-but-different math and an unvalidated baseline. | Each timing row references input and validation IDs; verifier checks contract equality, byte digests, pass state, and temporal order. |
| N2 | The numerical contract **MUST** bind formats, precision flags/readback, arithmetic and semantic references, elementwise envelope, nonfinite policy, distributions/seeds, and determinism policy. | Ambient precision drift, weak references, and lucky validation inputs. | Exact contract schema; environment probe and retained reference/input attachments recompute validation records. |
| N3 | Any elementwise or nonfinite failure **MUST** make that arm/cell timing-ineligible and **MUST NOT** be hidden by an aggregate metric. | Large local errors averaged away. | Mismatch/nonfinite counts and worst-element record gate `timing_eligible`; analyzer rejects timing IDs without a passing gate. |
| N4 | Cross-contract performance ratios **MUST NOT** be formed. | Speed claims between different accuracy or semantic work. | Analyzer requires identical `numerics_sha256` and semantic contract IDs for every contrasted row. |

### 3.7 GPU timing fairness

There is no context-free latency. Each timing stratum declares endpoint (`gpu_event` and/or synchronized host wall), synchronization points, stream/concurrency, eager or graph mode, allocation/workspace and kernel boundary, compile/cold/steady/amortized lifecycle, cache treatment, input rotation, warmup rule, requested and actual repetitions, randomized block design/order seeds, timeout/failure semantics, and observed device state.

`cold_l2`, `no_flush_static_input`, rotating-input, eager, graph, allocating, preallocated, floating-clock, and locked-clock results are distinct strata. A requested locked clock does not prove a stable observed clock. GPU evidence records physical device identity and telemetry sufficient to detect clock, power, thermal, utilization, and throttling differences. Unsupported clock control may use a frozen `floating_observed` stratum with telemetry and appropriately conditional scope.

Candidate and comparator execute on the same input bytes in randomized paired complete blocks, such as frozen AB/BA order. Raw per-block samples and actual counts are retained; p20/median/p80 alone are descriptions, not inferential units. Repeated launches are nested observations within a block.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| T1 | Every timing stratum **MUST** freeze timer, sync, stream, graph, allocation, lifecycle, cache/input, warmup/count, block/order, failure, hardware, and telemetry policies. | Asynchronous timing, hidden work, cache mismatch, and DVFS/order gaming. | Exact timing schema and environment probe; cross-arm equality and telemetry gate checks run before analysis. |
| T2 | Performance arms **MUST** use identical input bytes in randomized paired complete blocks and **MUST** retain raw samples, order, block IDs, and actual counts. | Input and order confounding, pseudoreplication, and quantile-only false precision. | Expected block cross-product, input digests, deterministic order regeneration, and summary recomputation from retained samples. |
| T3 | Every planned cell **MUST** have exactly one terminal outcome, including OOM, timeout, compile, numerical, or runtime failure. | Missing unfavorable measurements and partial favorable data. | Expected-cell set equality rejects missing, duplicate, nonterminal, and unexplained rows. |
| T4 | Timing strata **MUST NOT** be pooled or ratioed when cache, graph, timer, allocation, lifecycle, concurrency, input, or state gates differ. | Incomparable latency aggregation. | Contract ID and state-stratum equality checks reject the contrast. |
| T5 | A GPU-population claim **MUST** sample the declared GPU inferential unit; one instance **MUST** remain one-instance conditional evidence. | Launch replication being mistaken for device replication. | Hardware UUID/instance cluster counts and resampling hierarchy reject population wording without sampled outer units. |

### 3.8 Analysis, statistical units, and stopping

Each claim freezes an estimand before collection: endpoint, units, transform and direction, candidate-comparator contrast, target population or fixed census, sampling frame, aggregation/weights, missing/failure contribution, and hierarchy. Possible levels include launch within block, block within session, session within instance, instance within SKU, policy seed, workload within family, and target/source domain. The inferential unit is the level actually sampled for the stated population.

Timing arms are paired within randomized blocks. Resampling or modeling respects the frozen hierarchy; it never resamples below the declared inferential unit to manufacture precision. A policy-seed Student-t interval may quantify Monte Carlo policy randomness conditional on a fixed matrix, corpus, archive, and campaign, but not timing, workload, family, session, GPU, or SKU uncertainty. When outer units are absent or too few for the chosen estimator, those population levels are nonidentifiable and the output is conditional or descriptive.

A practical margin `delta`, direction, `alpha`, hypothesis family and ordering, multiplicity procedure, estimator/interval, and stopping policy are frozen. Fixed-size designs stop at the fixed count. Adaptive designs use a frozen confidence sequence or other time-uniform procedure and retain every look. Ordinary fixed-time confidence intervals are not valid after peeking. Unadjusted per-budget curves are descriptive.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| A1 | Every claim-bearing study **MUST** freeze a complete estimand and statistical hierarchy before dispatch. | Changing the question and pseudoreplication. | Estimand AST/digest, unit IDs/counts, formula, weights, population, and freeze timestamp; analyzer checks data levels against the AST. |
| A2 | Paired observations **MUST** remain paired, and uncertainty **MUST** resample/model no lower than the declared inferential unit. | Lost blocking efficiency and artificially narrow intervals. | Block-key equality and estimator metadata; verifier rejects a resampling level below the claim's inferential unit. |
| A3 | An inferential decision **MUST** use a pre-frozen practical margin, alpha, multiplicity family/procedure, and stopping rule. | Trivial wins, familywise cherry-picking, and optional stopping. | ClaimSpec and look ledger; deterministic decision regeneration applies family adjustment or confidence-sequence boundaries. |
| A4 | Every adaptive look **MUST** be retained, and an ordinary fixed-time interval **MUST NOT** support a claim after outcome-dependent peeking. | Selective stopping at a favorable estimate. | Look-ledger completeness and stopping enum; interval-method compatibility validator rejects the claim. |
| A5 | Policy-seed uncertainty **MUST** be labeled conditional on its fixed inputs and **MUST NOT** support workload-, timing-, family-, GPU-, or SKU-population wording. | Monte Carlo variability being inflated into empirical generalization. | Sampling-unit enum and conditioning digests are matched against the claim population and rendered limitations. |
| A6 | Multiple planned claims **MUST** belong to a frozen family or be marked descriptive. | Unadjusted multiple testing. | Family membership/order set equality and multiplicity-method replay; unmatched claims are downgraded to `descriptive`/`not_tested`. |

### 3.9 Execution semantics

The execution object binds the semantic envelope, expected logical calls and cells, deterministic seeds, environment/hardware predicates, source/dependency/image identity, deadlines, failure contribution, retry policy, abort/cancel behavior, and paid plan. A backend may add transport metadata but cannot alter semantic cell contents.

A local exploratory run and a strict remote run therefore differ only where the resolved protocol says they differ: coverage prefix, wall/cost bound, executor/backend, hardware predicate, and transport. A confirmatory run is not a longer continuation of exploratory observations; it begins from a separately frozen protocol and fresh eligible evidence.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| X1 | Local and remote execution **MUST** compile the same semantic call-plan hash for the same protocol. | Backend-specific benchmark semantics. | Cross-executor compile fixture compares workload, arm, numerics, timing, analysis, seed, and failure-plan digests. |
| X2 | Runtime hardware, environment, source, dependency, wheel, and image identities **MUST** satisfy frozen predicates and bind every row. | Fleet relabeling and mutable-runtime substitution. | Pre-allocation runtime probe, package/image digest closure, and row-to-environment digest checks. |
| X3 | An executor **MUST NOT** relax correctness, shorten warmup/repetitions, change cache behavior, drop failures, or choose arms based on observed outcomes. | A cheaper backend silently changing the scientific question. | Returned envelope digest, actual-count gates, plan/row set equality, and exposure/order audit. |
| X4 | Abort **MUST** stop new dispatch, request cancellation, reconcile every started call, charge all work, preserve partial evidence, and seal an outcome. | Runaway work, orphan calls, hidden cost, and replaced failures. | Journal transition and provider reconciliation checks require terminal/cancelled/orphaned state and cost entry for every attempt. |

## 4. Five-minute exploratory child protocol

The five-minute path is an immutable child of a `RESOLVED` parent. It carries `evidence_class: exploratory` and `parent_protocol_sha256`. Resolution selects a deterministic order-seeded prefix of the parent's planned cells and reduces only the declared cell count and wall/cost budget. It does not change a selected cell's candidate/comparator, input seed, numerical tolerance, warmup, repetition rule, cache treatment, timer, block structure, failure semantics, or raw-record format.

The executor stops dispatching when the five-minute hard cap is reached, requests cancellation for work that cannot finish within policy, retains every terminal and partial attempt record, seals incomplete coverage truthfully, and emits descriptions only. Even a complete lucky prefix cannot be relabeled, promoted, appended to a confirmatory bundle, or used to choose an untouched-looking claim under the same revision. Findings may inform a new draft whose relationship is recorded, but final evidence starts after a new freeze.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| Q1 | A five-minute run **MUST** be an `exploratory` child with its resolved parent digest and a hard monotonic wall-time cap. | An informal pilot being represented as final collection. | Parent reference resolution, evidence-class check, persisted start/deadline, and dispatch timestamps. |
| Q2 | The child's only semantic differences **MUST** be a deterministic cell prefix and reduced wall/cost budget; executor and transport metadata **MUST NOT** alter the inherited semantic envelope. | A “quick” path weakening correctness or timing to look faster. | Parent/child semantic diff allowlist contains only coverage/order-prefix and wall/cost-limit fields; compiled semantic envelope hashes remain equal for every retained cell. |
| Q3 | Prefix order **MUST** be deterministic and frozen before observations. | Choosing favorable early cases after results. | Recompute order from frozen seed and compare every dispatched cell index. |
| Q4 | Timeout **MUST** retain attempts and terminal rows, then seal incomplete coverage with descriptive output only. | Partial favorable data and vanished work. | Journal/coverage reconciliation plus claim-class validator. |
| Q5 | Exploratory observations **MUST NOT** be promoted, pooled into confirmatory evidence, or used as its evaluation rows. | Double use of pilot data and false preregistration. | Bundle ancestry/input-digest traversal rejects any exploratory ancestor or row in confirmatory claim inputs. |

## 5. Claim model and language

Every analyzer output has exactly one `kind`:

- `descriptive`
- `superiority`
- `inferiority`
- `noninferiority`
- `equivalence`
- `scoped_exhaustive_dominance`
- `transfer_benefit`

and exactly one `decision`:

- `supported`
- `not_supported_inconclusive`
- `stopped`
- `not_tested`

It repeats the candidate, comparator and optional reference roles; evidence class; estimand and direction; fixed-census or population scope; margin; alpha; interval/model; multiplicity family; stopping rule; provenance tier; input and analyzer digests; and limitations.

For an effect oriented so higher is better:

- superiority is supported only when the simultaneous lower bound is greater than `delta`;
- inferiority is supported only when the simultaneous upper bound is less than `-delta`;
- noninferiority is supported only when the simultaneous lower bound is greater than `-delta`;
- equivalence is supported only by two one-sided tests and the corresponding `100 × (1 - 2 alpha)%` interval lies strictly inside `[-delta, delta]`;
- otherwise the decision is `not_supported_inconclusive` at the named margin, not evidence of equality.

`scoped_exhaustive_dominance` applies only when the complete frozen evaluation-bank oracle for the enumerated candidate space loses under the same fixed corpus and protocol. It says nothing about unenumerated kernels, other spaces, other hardware, or a universal ceiling. `transfer_benefit` also satisfies the split and 2×2 design rule in §3.5.

Forbidden claim language includes “no difference,” “the same,” “equivalent,” “optimal,” “hardware ceiling,” “universally faster,” “generalizes to GPUs/workloads,” and “production improvement” unless the corresponding typed claim, margin/design, scope, and sampling frame support the exact phrase. A failed superiority test is reported as “superiority was not supported; the result is inconclusive at delta,” not as equality. Ratios always name numerator and denominator; “speedup” without direction is invalid.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| CL1 | Every published performance sentence **MUST** be rendered from a typed claim with the exact taxonomy, decision, roles, scope, direction, margin, and limitations. | Free-text overclaiming and ambiguous ratios. | Report renderer accepts ClaimSet objects only; phrase lint and report-to-claim ID/digest closure reject detached prose. |
| CL2 | A supported inferential claim **MUST** satisfy its frozen bound, margin, multiplicity, stopping, unit, completeness, and provenance rules. | A point estimate or nominal interval becoming a false win. | Decision engine regenerates bounds and eligibility from raw inputs and refuses inconsistent `supported`. |
| CL3 | An unsupported result **MUST** use `not_supported_inconclusive` and **MUST NOT** use equality or “no difference” language. | Absence of evidence being reported as evidence of absence. | Decision/phrase validator rejects forbidden wording unless a supported equivalence claim exists. |
| CL4 | A scoped exhaustive-dominance claim **MUST** close the complete frozen evaluation oracle and **MUST** state corpus, space, comparator, hardware, and protocol scope. | A bounded negative search result becoming universal optimality. | Oracle cell-set equality and required scope fields in renderer. |
| CL5 | Every ratio **MUST** name numerator and denominator and preserve a frozen direction. | Inverted “speedup” interpretation. | Ratio AST and report formatter print both roles; recomputation checks the stated direction. |

## 6. Paid execution, attempts, retry, cost, and abort

A paid plan freezes currency and tariff snapshot; maximum logical and physical calls, GPU-seconds, wall time and spend; approval identity; per-call timeout; idempotency namespace; retry classification; abort threshold; cancellation behavior; and treatment of unavailable billing. The cost estimand says whether compilation, search, source collection, failed attempts, storage, and amortization are included.

Before provider action, the collector durably records an intent containing a logical-call ID and idempotency key. Physical attempts record spawn, provider call ID, start, retrieval, terminal state, cancellation/reconciliation, and billing. Restart recovery queries or retrieves the recorded call; it does not spawn a replacement logical call.

The generic bundle attempt journal has an additive, canonical chain mode. Let $H_0$ be the lowercase SHA-256 of the empty byte string, `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. For transition $i$, serialize exactly the closed fields `cell_id`, `predecessor_sha256`, and `status`, in that order, as compact strict JSON followed by one LF; `predecessor_sha256` is $H_{i-1}$, and $H_i$ is the lowercase SHA-256 of those exact row bytes. The empty journal therefore has head $H_0$. Reordering, truncation relative to the root head, predecessor discontinuity, noncanonical JSON, CRLF, a missing final LF, blank lines, unknown fields, unknown cells, or a duplicate/invalid transition fails verification. This internal hash chain detects inconsistency within inventoried bytes; it is not a signature, provider attestation, or authentication.

The v1 default is one physical attempt. An optional retry policy permits only a frozen provider-classified pre-measurement infrastructure failure, with identical protocol bytes and resources. Every physical attempt and cost remains present and contributes according to the frozen policy. A benchmark error, numerical failure, timeout after measurement begins, OOM, unfavorable result, unresolved call, or identity mismatch is not retryable.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| P1 | Paid execution **MUST** freeze tariff/currency, call/GPU-time/wall/spend bounds, approval, timeout, abort rule, idempotency, retry policy, and cost estimand before dispatch. | Runaway spend and outcome-dependent cost accounting. | Pre-dispatch bound calculation, exact paid-plan schema, approval digest, and adapter-side hard limits. |
| P2 | Durable intent and idempotency identity **MUST** be recorded before any provider action. | Crash-window duplicate calls and untracked spend. | Fsync/durable-object acknowledgement timestamp precedes spawn; reconciliation queries the recorded key before further action. |
| P3 | Every physical attempt, provider retry, cancellation, orphan, and charge **MUST** remain in a hash-chained journal with a terminal or explicitly unresolved state. | Lucky retries and journal rewriting. | Predecessor-hash validation, unique provider IDs, attempt-count reconciliation, and external/root binding of chain head. |
| P4 | A retry **MUST** be predeclared, provider-classified as pre-measurement infrastructure failure, use identical bytes/resources, and retain every attempt/cost; all other failures **MUST NOT** be retried. | Replacement of bad measurements and semantic drift. | Retry-policy enum, failure-stage/provider classification, digest equality, max-attempt count, and ledger closure. |
| P5 | Actual cost **MUST** reconcile all completed, failed, cancelled, provider-retried, orphaned, and unresolved work against the approved bound. | Hidden failure spend and misleading cost efficiency. | Journal-to-bill set reconciliation and `actual <= approved`; unavailable itemization records `unknown_reason` and blocks cost-efficiency claims. |
| P6 | An abort **MUST NOT** dispatch new work or erase partial evidence. | Budget overrun and survivor-only publication. | No spawn timestamp may follow abort; cancellation/reconciliation and sealed artifact inventory cover all prior intents. |

## 7. EvidenceBundle, provenance, publication, and retention

### 7.1 Root and closure

A `heliostune.bundle/1` root is strict JSON and contains the authoritative schema-defined form of:

```text
schema: literal "heliostune.bundle/1"
bundle_id: nonblank string
created_at: RFC 3339 string
protocol: {path: relative path, sha256: digest, bytes: integer >= 1}
lifecycle: {
  state: DRAFT | RESOLVED | FROZEN | DISPATCHED |
         SEALED | VERIFIED | ANALYZED | PUBLISHED,
  outcome: pending | completed | failed | aborted
}
attempts: {
  path: relative path, sha256: digest, hash_chain_head: digest,
  logical: integer >= 0, physical: integer >= 0,
  terminal: integer >= 0, orphaned: integer >= 0
}
coverage: {
  expected_cells: integer >= 0, terminal_cells: integer >= 0,
  successes: integer >= 0, failures: integer >= 0
}
artifacts: [{role: string, path: relative path, media_type: string,
             bytes: integer >= 0, sha256: digest}]
provenance: {
  attestation: none | self_attested_backend | provider_signed,
  offline_reproduction: not_checked | partial | complete
}
signatures: [{scheme: string, signer: string,
              subject_sha256: digest, signature: string}]
```

Protocol, attempts, and artifact paths are mutually unique; artifact roles and signatures are unique. Before `SEALED`, lifecycle outcome is exactly `pending`; at `SEALED` and later it is exactly one of `completed`, `failed`, or `aborted`. Count invariants include `terminal <= logical`, `orphaned <= physical`, `terminal_cells <= expected_cells`, and `successes + failures == terminal_cells`. An incomplete bundle remains representable so failed or aborted exploratory prefixes can seal truthfully; completed bundles and both strict evidence tiers require exact expected/terminal cell-set equality. The root field maps above remain unchanged: the controls below are additive artifacts, not new bundle-root fields.

Every bundle binds the protocol digest roles `plugin`, `workloads`, `candidates`, `comparators`, `splits`, `numerics`, `timing`, `analyzer`, `expected_cells`, `environment_predicate`, and `failure_policy`. It also binds `paid_plan` and `parent_protocol` exactly when the corresponding protocol digest is non-null. Missing, duplicate, unexpected optional, or digest-mismatched bindings fail. `expected_cells` and `terminal_cells` are strict JSON arrays of unique nonblank cell-ID strings. Their parsed lengths must equal every self-declared protocol, coverage, and attempts count that they govern. A failed or aborted exploratory terminal list may only be a prefix of its expected list.

Full plugin → suite custody uses reserved flat artifact roles and paths. In the plugin's `suite_refs` order, index $i$ is role `plugin_suite_<i>` at path `plugin_suite_<i>.json`; indices are contiguous from zero. Role `selected_suite` at `selected_suite.json` is an exact closed descriptor with only schema literal `heliostune.selected-suite/1` and nonnegative integer `plugin_suite_index: i`. The selected index binds the inventoried suite the bundle declares selected; it is not a mutable path, an independently asserted identity, or proof of backend execution. For every reference, verification uses only the inventoried bytes and checks its digest, suite ID and revision, its plugin ID and canonical decimal plugin version back-reference, and the suite's strict schema. The protocol plugin ID, version, and digest must match that inventory, while the plugin's ordered domains and arm IDs must equal the first-seen aggregation across the suites in reference order.

Every plugin-suite inventory entry and descriptor uses media type `application/json`. The selected-suite descriptor payload uses canonical strict JSON: two-space indentation, lexicographically sorted keys, and exactly one trailing LF.

These roles are an explicit opt-in. If no plugin-suite reserved role is present, a structurally valid legacy v1 bundle remains parseable and reports `plugin_suite_custody: not_checked`; it gains no custody claim. Presence of any role beginning `plugin_suite` or `selected_suite` selects strict mode, so a malformed prefix, missing descriptor or index, gap, extra suite, wrong path, malformed descriptor, identity/back-reference mismatch, or incomplete aggregate fails closed rather than falling back to legacy behavior.

The required attempts path is strict JSONL. Chained mode is selected by role `attempt_chain` at `attempt_chain.json`, whose exact closed descriptor value is `{"schema":"heliostune.attempt-chain/1"}`. Its payload uses the same two-space-indented, sorted-key strict JSON with one trailing LF. Presence of any `attempt_chain`-prefixed role selects strict handling, and every form except that one exact role, path, media type, and payload fails closed. In chained mode every row uses the three-field canonical predecessor algorithm in §6 and the root `hash_chain_head` must equal the computed final head, including $H_0$ for an empty journal. With no such prefix, only the legacy exact two-field `{cell_id,status}` rows are accepted and `attempt_journal_hash_chain` is `not_checked`; mixed or partially opted-in forms fail. In either mode, each cell begins at `pending`, may proceed to `running`, and terminates once at `success` or `failure`, and journal terminal IDs and success/failure counts are bound to `terminal_cells` and coverage.

Verification of an arbitrary bundle opens the resolved bundle directory once and then opens every normalized relative path component descriptor-relatively with symlink following disabled. It requires each opened target to be a regular file, hashes and parses the bytes read from that same descriptor, and rejects repeated `(st_dev, st_ino)` identities, including hard-link aliases between declared roles. Producers call `verify_bundle_v1_from_directory_fd` with their already-open staging directory descriptor; the verifier duplicates it, and both pre-rename and post-rename verification operate solely through that pinned descriptor. Any supplied directory path is diagnostic text only, never path-resolution authority. These containment checks prevent path substitution and aliasing; they do not authenticate the bytes' author.
Reads are finite before allocation: a bundle root is limited to 2 MiB, every
protocol/attempt/artifact component to 32 MiB, and one verification or selected
artifact capture to 64 MiB in aggregate. Declared and descriptor-reported sizes
are checked before payload reads, growth past the bound fails during the read,
and descriptor size/timestamp identity must remain stable across it. Larger
evidence must be compressed or split into a separately specified format rather
than weakening the verifier.

`verify_bundle_v1` establishes structural closure only, not publication eligibility. It validates strict root/protocol schemas, descriptor-contained non-escaping unique paths and roles, byte counts and digests, protocol digest-role closure, lifecycle phase/outcome compatibility, strict cell identities, coverage sets/counts, the selected strict or legacy attempt transitions, and—when opted in—the complete transitive plugin inventory and canonical predecessor chain. A control is `checked` only after its applicable verification succeeds. No reserved plugin-suite roles and no attempt-chain descriptor retain their respective `not_checked` legacy statuses. Attempt reconciliation is `checked` only when rows evidence every final logical state, retry policy is `none`, `max_physical_attempts` is one, physical equals logical, and orphaned is zero; retry/orphan/provider cases remain `not_checked`. A chained, sealed, nonempty journal cannot end in a live state; an empty aborted pre-dispatch journal remains valid. Protocol ancestry, exploratory nonpromotion, semantic-content validation beyond digest identity, claim eligibility, analyzer replay, provenance-tier derivation, signature cryptography, catalog membership, provider retry/billing truth, and complete offline reproduction remain `not_checked`. `publication_eligible` remains false.

### 7.2 Canonical VerificationRecordV1

After structural verification succeeds, issue #32's CPU-only
`VerificationRecordV1` gives that exact result durable deterministic bytes. It
is an additive description of what this verifier checked, not a mutation of
`heliostune.bundle/1`. Its exact closed wire shape is:

```text
schema: literal "heliostune.verification-record/1"
verifier: {
  package: string, version: string, source_sha256: digest,
  sources: [{path: relative path, bytes: integer >= 0, sha256: digest}]
}
bundle: {
  schema: literal "heliostune.bundle/1", bundle_id: string,
  root: {bytes: integer >= 0, sha256: digest},
  protocol: {
    path: relative path, bytes: integer >= 0, sha256: digest,
    study_id: string, revision: integer >= 1
  },
  attempts: {
    path: relative path, bytes: integer >= 0, sha256: digest,
    hash_chain_head: digest
  },
  artifacts: [{
    role: string, path: relative path, media_type: string,
    bytes: integer >= 0, sha256: digest
  }]
}
lifecycle: {state: lifecycle state, outcome: lifecycle outcome}
evidence_class: evidence class
controls: {
  protocol_ancestry: status,
  evidence_nonpromotion: status,
  semantic_content_beyond_digests: status,
  plugin_suite_custody: status,
  attempt_journal_hash_chain: status,
  attempt_reconciliation: status,
  claim_eligibility: status,
  analyzer_replay: status,
  provenance_tier_derivation: status,
  signature_cryptography: status,
  catalog_membership: status,
  offline_reproduction: status
}
claim_eligible: boolean
publication_eligible: boolean
```

The status enum is closed to `checked | not_checked | not_applicable | failed`.
`checked` means this verifier completed the named control successfully;
`not_checked` means it did not perform the control; `not_applicable` records
that the control does not apply; and `failed` records a control that did not
pass. The eligibility rule is deliberately stricter than applicability:
`all_checked` means every one of the twelve exact controls is `checked`, and
both `claim_eligible` and `publication_eligible` must equal `all_checked`.
Consequently `not_checked`, `not_applicable`, or `failed` anywhere forces both
booleans false. Lifecycle state, evidence class, and provenance never alter
this formula. In particular, `VERIFIED`, `ANALYZED`, and `PUBLISHED` labels do
not confer verification or eligibility. `has_failed_controls` is an in-memory
convenience only and is not a wire field.

Canonical encoding is two-space-indented, lexicographically sorted-key strict
JSON with exactly one trailing LF. Artifact entries are sorted by `(role,
path)`. The verifier source roster and order are fixed and lexicographic:
`heliostune/_offline_worker.py`, `heliostune/_reference_analyzer.py`,
`heliostune/artifacts.py`, `heliostune/errors.py`,
`heliostune/methodology.py`, `heliostune/offline_replay.py`,
`heliostune/scope.py`, `heliostune/validation.py`, and
`heliostune/verification.py`. For the aggregate
SHA-256, initialize the hash with
`b"heliostune.verification-sources/1\0"`; for each source in that order append
the UTF-8 path length as an eight-byte big-endian integer, the path bytes, the
source byte count as an eight-byte big-endian integer, and the raw 32-byte file
digest. The installed resources, package version, and aggregate are captured
at module import and recaptured when a record is built; any mismatch or
unreadable source fails. This is descriptive self-identification, not
authentication.
New records emit this nine-file roster. To preserve the schema's historical
loader contract, decoding also accepts exactly issue #32's earlier six-file
roster (`artifacts.py`, `errors.py`, `methodology.py`, `scope.py`,
`validation.py`, and `verification.py`, with their full package-relative
paths). No other roster or order is accepted, and historical bytes are never
rewritten into the current identity.

The canonical record is location-free: it includes only bundle-relative
protocol, attempts, and artifact paths and excludes the absolute bundle path,
runtime path, output path, timestamp, hostname, PID, executable, and random
identifier. Loading requires byte-for-byte canonical re-encoding. Building a
base record does not perform replay. Neither a base record nor the successfully
upgraded replay record is a signature, authentication, provider truth, a
semantic or statistical correctness result, a claim promotion, or full
dependency/campaign reproduction.

#### 7.2.1 Audited CPU offline analyzer replay

An opted-in replay bundle inventories exactly one `analyzer` artifact with
media type `application/json`, whose digest is bound by
`protocol.analysis.analyzer_sha256`. Its bytes load canonically as the closed
`AnalyzerManifestV1` shape:

```text
schema: literal "heliostune.analyzer-manifest/1"
analyzer_id: string
runner_api: literal "heliostune.offline-replay/1"
implementation: {
  source_sha256: digest,
  sources: [{role: string, media_type: string, bytes: integer >= 0, sha256: digest}]
}
inputs: [{role: string, media_type: string, bytes: integer >= 0, sha256: digest}]
outputs: [{role: string, media_type: string, bytes: integer >= 0, sha256: digest}]
representation: literal "byte_exact"
```

The binding entries above are `AnalyzerArtifactBindingV1`; the implementation
object is `AnalyzerImplementationV1`. Objects have exact fields and JSON scalar
types. Source, input, and output bindings are nonempty ordered lists with
nonempty roles; roles are unique and disjoint across all three lists. Encoding
is two-space-indented, sorted-key strict JSON with exactly one trailing LF, and
the loader requires byte-identical canonical re-encoding. The manifest contains
no path, command, import/module name, or entrypoint. Its analyzer ID must name a
built-in audited registry
entry, and the manifest's implementation source roster, role/media-type
contracts, bytes, and digests must exactly match that static registry entry and
the inventoried bundle artifacts. Artifact bytes never select executable code.

The initial registry contains only
`heliostune.reference.integer-summary/1`. Its implementation has sole binding
role `analyzer_source` with media type `text/x-python`; the registry privately
maps that identity to the audited installed
`heliostune/_reference_analyzer.py`, while captured bundle source bytes are
compared but never executed. The callable accepts ordered `(role, bytes)`
tuples with sole input role `analysis_input` and media type `application/json`;
it emits sole output role `analysis_summary` with the same media type. Input is
exact canonical JSON whose sole field `values` is an array of 1–4096 signed,
exact JSON integers.
Output is canonical JSON binding the input SHA-256, count, minimum, maximum,
and sum. The callable uses only already-loaded pure functions: it imports
nothing and performs no external action during invocation.

The parent runner first requires the base record's `plugin_suite_custody`,
`attempt_journal_hash_chain`, and `attempt_reconciliation` statuses to be
`checked`; caller-supplied statuses are never trusted. It opens the resolved
bundle directory without following symlinks, requires its retained
device/inode identity to match, and captures the manifest, implementation
sources, declared inputs, and committed outputs through the same bounded,
descriptor-relative methodology reader with byte-count and SHA-256 checks. It
reverifies the same open bundle directory descriptor after capture and again
after both replay runs.

Each run receives a distinct empty workspace. The fixed absolute launch prefix
is `/usr/bin/setpriv --no-new-privs /usr/bin/unshare --user --map-root-user
--net --mount --pid --fork --kill-child=SIGKILL --mount-proc`, followed by the
absolute current Python executable and
`-B -P -s -m heliostune._offline_worker`. The child receives only `HOME=/`,
`LC_ALL=C`, `LANG=C`, `TZ=UTC`, `PYTHONHASHSEED=0`, and
`PYTHONDONTWRITEBYTECODE=1`. There is no fallback when user/network/mount/PID
namespace creation, the tmpfs remount, chroot, or another isolation
precondition is unavailable.

The parent sends one bounded canonical base64 request on standard input and
captures bounded stdout/stderr in regular files. Timeout and communication
failure handling kills and reaps the whole process group. The request binds the
exact manifest implementation plus the parent's complete
`VerifierIdentityV1`. Before replay, the child independently captures its
installed package version and nine-file verifier source identity and requires
exact equality, preventing a different package resolved by the child
interpreter from silently performing the run.

The worker preloads only the fixed registry and reads the request, then requires
PID 1, effective UID/GID 0, one-ID user/group mappings, and `NoNewPrivs: 1`. It
mounts a fresh empty mode-0555 tmpfs with `nosuid,nodev,noexec`, bind-remounts
it read-only, re-enters that mount, and requires both `ST_RDONLY` and an `EROFS`
write probe before chrooting and changing to `/`. It closes every non-stdio
descriptor, applies bounds to request/output size, CPU, address space, file
size, open descriptors, and process count, and installs a deny-and-latch Python
audit hook before invoking the selected fixed callable. Attempts to open or
import, create sockets or resolve DNS, launch subprocesses, fork, exec, call
`os.system`, use `ctypes`, or add an audit hook fail the run even if analyzer
code catches the immediate exception. User/network namespaces plus the
read-only empty tmpfs chroot and resource/process bounds are the primary
sandbox; the audit hook is a tripwire, never the sandbox claim.

The parent rejects timeout, nonzero exit, any stderr, malformed, trailing, or
oversized result bytes, output role/count/order disagreement, unequal run
outputs, or output bytes that differ from the pre-captured committed artifacts.
Only two successful runs in distinct workspaces whose outputs are pairwise
byte-identical and equal the committed bytes produce `OfflineReplayResult`.
That result retains the original `VerifiedBundle`, manifest, identities for
both runs, and an upgraded record.

The replay-specific
`build_replay_verification_record_v1(result: OfflineReplayResult)` accepts only
the success-only runner result—never a caller-supplied record or statuses. It
reconstructs the base record and changes exactly `analyzer_replay` and
`offline_reproduction` to `checked`; every other control, bundle identity,
lifecycle value, and evidence-class value is identical, and eligibility is
recomputed from all twelve controls. The file writer likewise accepts
`(path, result)` and validates that exact replay result before calling the safe
record publisher. Neither path changes the `VerifiedBundle` limitations. This
`offline_reproduction` control means only a
successful same-host drill of the registered analyzer and declared committed
derived bytes. It does not establish authenticity, cross-host bit
reproducibility, provider truth, semantic or statistical correctness, GPU
recollection, claim eligibility or promotion, or complete software dependency
or campaign reproduction. Replay downloads nothing, invokes no GPU or backend,
and authorizes no paid call; the maximum spend is **$0**.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| E1 | A bundle **MUST** close its protocol, attempts, expected cells, raw and failed observations, provenance, costs, analyzer, claims, reports, and required attachments by path, byte count, role, media type, and digest. | Selective omission, dangling inputs, and mixed evidence generations. | Recursive inventory/reference traversal, digest/size recomputation, and required-role set equality. |
| E2 | Sealing **MUST** permanently close attempts and account for exactly one terminal outcome per planned cell or a protocol-permitted incomplete exploratory prefix. | Late appended retries and favorable partial datasets. | Seal event fixes journal head; expected/terminal set equality and evidence-class coverage rules reject later records. |
| E3 | Verification **MUST** fail closed on any schema, reference, lifecycle, custody, coverage, semantic, replay, claim, or policy error. | Warning-only publication of invalid evidence. | VerificationRecord contains one of the four closed statuses for every control and sets both eligibility booleans true only when all twelve controls are `checked`. |
| E4 | Derived summaries, claims, and reports **MUST** regenerate offline from bound inputs and the frozen analyzer. | Hand-edited results and network-dependent analysis. | Registry-only network-disabled replay runs twice and compares every declared `byte_exact` output pair and committed artifact byte-for-byte. |

### 7.3 Provenance tiers

Execution provenance is one of:

- `none`: no execution assertion beyond bound artifact bytes;
- `self_attested_backend`: an operator/backend assertion is bound, but no independently verifiable provider receipt exists;
- `provider_signed`: the configured trust policy verifies a provider or independent receipt over the envelope, function/image, hardware, call ID, timestamps, nonce/idempotency key, and output digest.

Offline reproduction provenance is separately `not_checked`, `partial`, or `complete`. `complete` means a verifier can work without network access, validate the retained software closure, install it, replay analysis/report generation, and reproduce derived evidence bytes. It does not mean GPU measurements can be recollected without equivalent hardware. Missing redistributable dependencies, base image closure, build inputs, or byte-identical analysis replay yields `partial` or failure, never an inferred `complete`. The issue-#33 record control named `offline_reproduction: checked` reports only its bounded same-host registered-analyzer drill; it does not infer this broader complete provenance tier.

Provider signatures, a seven-year retention period, and two independently administered replicas are policy-profile controls rather than universal v1 requirements. A strict publication profile may require them. Lower truthful tiers remain representable but cannot support authentication, durability, or full-offline-reproduction wording.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| PR1 | A bundle **MUST** report exact attestation and offline-reproduction tiers and **MUST NOT** claim a stronger tier when a receipt or retained closure is missing. | Self-asserted execution and incomplete custody being marketed as authenticated/reproducible. | Trust-policy signature/receipt validation and network-disabled closure drill derive, rather than trust, both enums. |
| PR2 | `provider_signed` **MUST** bind envelope, function/image, hardware, provider call, timestamps, nonce/idempotency key, and output digest under the configured trust policy. | Fabricated, replayed, or relabeled remote payloads. | Signature, identity, time, revocation/transparency where applicable, nonce, and digest-link checks. |
| PR3 | `offline_reproduction: complete` **MUST** include retained source/build/dependency/image inputs and successful network-disabled derived-byte regeneration. | Registry mutation, dependency disappearance, and irreproducible reports. | Restore drill verifies all artifact digests, offline installation, analyzer replay, and report comparison. |

### 7.4 Catalog, atomic publication, and retention

The versioned StudyRegistry declares the complete expected study set. The public catalog contains every expected study and revision, including completed, failed, aborted, negative, withdrawn, private-tombstone, and superseded records. Publication stages a complete immutable generation and atomically advances one content-addressed root pointer; readers see either the prior valid generation or the next valid generation, never a mixture.

Each publication declares a retention policy ID, expiry or indefinite retention, required replica/administrator count, scrub schedule, restore-drill schedule, and withdrawal/legal-deletion behavior. A policy profile may impose stronger duration, signature, transparency, escrow, or replica controls. Legal deletion leaves a signed tombstone and changes affected claim verifiability; it does not silently rewrite the catalog.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| PUB1 | The catalog **MUST** equal the registry's complete expected study/revision set, including failures, negative results, withdrawals, tombstones, and superseded revisions. | Cherry-picked publication and invisible failed studies. | Registry/catalog study-ID set equality, uniqueness, required-role checks, and registry digest binding. |
| PUB2 | Publication **MUST** expose a complete verified generation atomically through a content-addressed root. | Crash-created mixed artifact, sidecar, summary, and report generations. | Staging verification followed by compare-and-swap/atomic rename; fault injection proves readers see only prior or next valid root. |
| PUB3 | A published bundle **MUST** declare and satisfy its named retention profile and **MUST** leave a signed tombstone for withdrawal or required deletion. | Silent evidence loss and unverifiable disappearance. | Retention manifest, expiry/replica/scrub/restore checks, and catalog tombstone validation. |

## 8. CLI lifecycle: local exploration to strict evidence

The currently implemented CPU-only inspection, record, and replay surface is:

```bash
heliostune verify-plugin PATH
heliostune verify-suite PATH
heliostune verify-bundle path/to/bundle/bundle.json
heliostune verify-bundle path/to/bundle/bundle.json --format json
heliostune verify-bundle path/to/bundle/bundle.json --output path/to/bundle.verification.json
heliostune replay-bundle path/to/bundle/bundle.json
heliostune replay-bundle path/to/bundle/bundle.json --format json
heliostune replay-bundle path/to/bundle/bundle.json --output path/to/bundle.replay-verification.json
heliostune list-scope
```

`verify-plugin` validates the strict plugin root and resolves its relative suite
paths and SHA-256 values. `verify-suite` validates one strict standalone suite.
With no output flags, both bundle commands retain human-readable text.
`verify-bundle` stops after structural verification and emits the base record;
`replay-bundle` emits only after the complete isolated two-run drill and exact
record upgrade succeed. `--format json` writes exact canonical
VerificationRecord bytes to standard output without Rich rendering. `--output
PATH` implies JSON and writes silently to a new sibling file through the
replay-specific exact-result validator when replaying. Its existing parent must
be the bundle directory's immediate parent and match the device/inode identity
captured through the pinned bundle descriptor; arbitrary destinations use
external JSON stdout redirection. The record/result must exactly match the
original `VerifiedBundle`, and encoding completes before output is touched.
Explicit `--format text --output PATH` is rejected before verification or
replay.

A structurally verified base record with deferred controls exits zero even
though its eligibility booleans are false. Replay exits zero only after both
runs equal one another and the committed output bytes. Any `failed` control or
verification, manifest, isolation, audit, replay, build, encode, or pre-link
write error exits 2 with no success bytes or destination. The bundle-parent
relationship is rechecked immediately before and after the irreversible
no-replace link. No topology immutability is claimed: a hostile rename can make
the requested pathname stale or unrecoverable after linking. A post-link error
reports committed/ambiguous state: the complete linked destination is not
rolled back, but directory durability may be ambiguous. Publication uses
unnamed `O_TMPFILE` storage and an unprivileged procfd source for atomic
no-replace `linkat`; it fails closed if either capability is unavailable.
Closing the unnamed fd performs cleanup.

`list-scope` reports the closed vocabularies and initial suite template IDs.
Replay establishes only same-host registered-analyzer reproduction of declared
committed derived bytes. These commands make no backend/GPU execution,
correctness, performance, signature/authenticity, provider retry/billing,
cross-host bit-reproducibility, semantic/statistical-truth, claim-promotion, or
full dependency/campaign-reproduction assertion.

The intended full evidence lifecycle is explicit rather than a mode flag that silently changes benchmark semantics:

```bash
heliostune plugin list
heliostune plugin describe PLUGIN
heliostune plugin validate PLUGIN

heliostune resolve study.json --output resolved.json
heliostune plan resolved.json --mode explore --wall-time 5m --output quick-protocol.json
heliostune run quick-protocol.json --executor local --output quick-bundle/
heliostune verify quick-bundle/
heliostune diff resolved.json quick-protocol.json

heliostune freeze resolved.json --output protocol.json
heliostune collect protocol.json --executor local|modal --output bundle/
heliostune seal bundle/
heliostune verify bundle/
heliostune analyze bundle/
heliostune report bundle/
heliostune publish bundle/ --registry REGISTRY --policy POLICY
heliostune verify-catalog CATALOG
```

The lifecycle commands in the preceding block describe the contract, not a claim that all of them exist today; they are distinct from the implemented declaration commands above. The prospective `plugin validate` step checks deterministic resolution and interface compatibility. `resolve` materializes every implicit choice. `plan --mode explore` creates the non-promotable child. `diff` displays the semantic allowlist. `freeze` assigns immutable identity. `collect` uses one envelope on either backend. `seal`, `verify`, and `analyze` are separate so collection cannot declare itself valid. `publish` checks a registry and policy profile before the atomic root switch.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| CLI1 | `freeze` **MUST** reject unresolved defaults and **MUST** display the canonical semantic diff from its parent/revision. | Invisible defaults and accidental scope change. | Resolver completeness schema and field-by-field digest-aware diff acknowledgement. |
| CLI2 | `collect` **MUST** accept a frozen protocol rather than editable study input. | Paid execution of a moving draft. | State/digest precondition and immutable protocol copy in the output bundle. |
| CLI3 | `verify`, `analyze`, and `publish` **MUST** enforce `SEALED`, `VERIFIED`, and `ANALYZED` preconditions respectively. | Collection self-certifying and unverified analysis reaching publication. | Lifecycle transition validator and canonical VerificationRecord/ClaimSet references. |
| CLI4 | Strict execution **MUST NOT** consume exploratory rows or continue an exploratory bundle. | A local pilot becoming part of final evidence. | Fresh bundle ID, ancestry/input-digest traversal, and class eligibility check. |

## 9. Legacy policy

Anything without the exact schema literal `heliostune.protocol/1` or `heliostune.bundle/1` is legacy. Legacy bytes and their original claims remain immutable and are interpreted only under their original protocol and known limitations. An importer may inventory a legacy artifact as `legacy_unverified`, recording original bytes, source identity, known fields, unknown fields, and field-loss report. It cannot invent target exposure history, raw timing samples, telemetry, attempts, numerical checks, margins, multiplicity, provenance, or stronger claim eligibility.

Current Parhelion policy-seed intervals remain conditional on their fixed matrices and campaigns; they do not become workload-, timing-, family-, session-, GPU-, or SKU-population intervals. Existing Hopper evidence remains a one-instance, fixed-corpus, same-bank engineering STOP; selection optimism is explicit and no superiority claim follows. Both studies remain legacy plugins rather than `heliostune.plugin/1` or `heliostune.suite/1` migrations. Neither case is backfilled into v1 eligibility.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| L1 | A legacy import **MUST** preserve original bytes and identity, label unknown facts as unknown, and remain `legacy_unverified`. | Manufactured provenance and retroactive methodological certainty. | Source digest comparison, import field-loss report, and hard-coded ineligibility for v1 claim promotion. |
| L2 | A v1 artifact **MUST NOT** overwrite, alias, or silently replace a legacy artifact or claim. | Historical revisionism and broken citations. | Path/digest immutability catalog rules and distinct study/schema identities. |
| L3 | Legacy observations **MUST NOT** gain v1 eligibility merely by being wrapped in a v1 bundle. | Schema laundering. | Provenance traversal detects legacy source roles and restricts them to attachments or explicitly scoped legacy analyses. |

## 10. Acceptance tests

A conforming implementation demonstrates at least these observable tests:

1. **Exploratory timeout:** a five-minute child times out mid-prefix, retains attempts and terminal rows, emits descriptions only, and promotion is rejected.
2. **Immutable revision:** changing any semantic byte after freeze produces a new digest/revision and `supersedes` edge; old evidence remains resolvable.
3. **Executor parity:** local and remote compilation produce identical semantic call-plan hashes while explicit transport and hardware predicates may differ.
4. **Leakage rejection:** reuse of an evaluation row, normalization fit, or connected leakage component makes a confirmatory claim ineligible.
5. **Closure failure:** missing, duplicate, forbidden-retry, orphaned, or nonterminal planned cells fail verification while preserving failure evidence.
6. **Numerical gate:** candidate or comparator numerical failure prevents timing eligibility and every cross-contract ratio.
7. **Timing replay:** raw paired blocks regenerate summaries deterministically; sample, order, cache, input, or telemetry tampering fails verification.
8. **Statistical decisions:** fixtures cover practical margins, TOST equivalence, multiplicity adjustment, optional stopping, inferential-unit mismatch, policy-seed conditioning, and one-GPU population rejection.
9. **Paid recovery and abort:** a crash after dispatch reconciles the original idempotency key without spawning a replacement; abort cancels/reconciles all work and accounts for every cost.
10. **Atomic complete publication:** fault injection exposes only the prior or next valid root; removing a registered negative/failed study, changing replay provenance, or elevating legacy evidence fails publication.
11. **Strict schemas:** duplicate/unknown keys, boolean-as-integer, nonfinite number, uppercase/short digest, escaping path, dangling reference, and invalid lifecycle transition are rejected.
12. **Claim language:** an inverted unnamed ratio, “no difference” after unsupported superiority, universal wording from a fixed census, and deployable wording for an evaluation oracle are rejected.
13. **Opt-in custody and attempt chain:** complete inventoried plugin-suite closure, selected-suite identity, canonical predecessor bytes, empty $H_0$, truncation/reorder rejection, legacy `not_checked`, descriptor/inode containment, and producer pre-rename postconditions are verified for new local/native bundles.
14. **Canonical verification record:** identical verified bundle bytes produce identical location-free record bytes; all four control statuses round-trip strictly, deferred or failed controls force both eligibility booleans false, lifecycle labels cannot promote them, and noncanonical input fails loading. Default output stays human-readable, canonical JSON uses exact stdout bytes, and sibling-only no-replace output rechecks the descriptor-identified bundle parent before and after linking. Pre-link failure creates no destination; a post-link failure reports committed/ambiguous state, does not roll back the complete linked entry, and acknowledges that hostile rebinding can make the requested pathname stale or unrecoverable.
15. **Audited offline analyzer replay:** strict canonical manifests reject unknown fields/types, paths, commands, entrypoints, registry/static-roster mismatch, and bound-byte/digest mismatch; the fixed integer-summary analyzer validates its exact canonical domain; namespace/chroot/audit attempts and unavailable isolation fail closed; two distinct read-only workspaces produce identical ordered bytes that also equal committed outputs; and the upgraded record changes only the two replay controls without promoting eligibility. CLI text/JSON/file modes emit only after success, and file failures retain the no-replace/committed-ambiguous semantics.

| ID | Normative requirement | Failure prevented | Machine enforcement |
|---|---|---|---|
| AT1 | A v1 implementation **MUST** pass every applicable acceptance test above before advertising that implemented surface as conforming. | Methodology branding without behavioral enforcement. | CI records the named test, implementation component, policy profile, and passing revision; absent components remain explicitly unimplemented. |

## 11. Implementation status

This table separates the active, partially implemented design target from repository reality. “Partial” is not v1 eligibility: each gap remains a fail-closed implementation boundary until its stated controls and acceptance tests pass.

| Surface | Current repository status | Consequence |
|---|---|---|
| Strict JSON and exact-type validation primitives | Implemented for existing artifact families; coverage varies by historical schema. | Useful building block, not proof of protocol/bundle closure. |
| `heliostune.protocol/1` and `heliostune.bundle/1` exact schemas | Strict frozen/slotted value objects and file loaders parse the unchanged field maps above. Generic bundle verification resolves bound bytes and structural custody controls. | Parsing a v1 root alone is not bundle closure, and no historical artifact is automatically a v1 artifact. |
| `heliostune.plugin/1` and `heliostune.suite/1` declarations | Strict structural loaders and standalone verification are implemented for the closed initial scope. Plugin verification and opted-in bundle verification share one in-memory transitive inventory check. | Checked plugin → suite closure establishes internal inventoried identity, not execution, authorship, or claim eligibility. |
| Generic study-plugin resolver and runtime contracts | Not implemented end to end. Existing studies use study-specific modules and manifests. | The full lifecycle plugin commands and interfaces remain unimplemented design surfaces; no generic local or remote backend exists. |
| Eight-state immutable lifecycle and revision registry | Not implemented end to end. | Existing manifests retain their own state models and legacy interpretation. |
| Local/remote canonical semantic envelope | Partial in study-specific collectors; no generic v1 executor contract. | Executor parity is not yet generally established. |
| GPU raw randomized paired blocks and full telemetry | Not implemented for published Parhelion/Hopper timing artifacts, which retain aggregate quantiles and limited state. | Those timings remain legacy, fixed-protocol evidence and do not support new timing-population inference. |
| Numerical gate for candidates and comparators | Partial and study-specific; published collectors do not establish the generic two-reference v1 contract for every arm. | Existing correctness statements keep their frozen historical scope. |
| Attempt chain and reconciliation | The generic canonical predecessor chain is implemented as an opt-in control. New local/native bundles emit it and no-retry reconciliation can be `checked`; legacy two-field journals remain parseable with chain `not_checked`. Provider retry adjudication, provider physical-attempt truth, billing, and complete cost reconciliation remain unimplemented generically. | A checked internal chain/reconciliation result is neither a provider receipt nor proof of cost, retries, authorship, or authenticity. |
| Typed claim taxonomy and fail-closed generic analyzer | `ClaimSpec` parsing and class-level eligibility checks are implemented; analyzer output decisions, statistical replay, and generic report rendering are not implemented end to end. Historical reports still use study-specific models and prose. | A valid ClaimSpec is not a supported result, and historical claims are not silently rewritten. |
| EvidenceBundle custody, verification record, and publication | Generic structural closure, descriptor-contained file reads, opted-in plugin → suite custody, and opted-in attempt chaining are implemented. New local/native producers emit the complete reserved inventory and descriptors, require custody, chain, and no-retry reconciliation to be `checked`, and verify staging before atomic no-replace rename. Issue #32 adds a deterministic, location-free `heliostune.verification-record/1` with nine-file descriptive verifier identity, exact bundle identities, all twelve control statuses, strict eligibility booleans, canonical loading, exact JSON stdout, and descriptor-pinned sibling-only no-replace output with explicit committed/ambiguous post-link reporting. Signature/authenticity checks, complete dependency/campaign reproduction, full registry/catalog closure, and the complete v1 publication transaction remain unchecked or unimplemented. | Base records retain deferred controls and are not claim- or publication-eligible. Internal closure and a record are not provider truth, authenticity, semantic or statistical correctness, or full reproduction; existing publication workflows remain legacy rather than a v1 conformance claim. |
| Audited CPU analyzer replay | Issue #33 implements strict `heliostune.analyzer-manifest/1`, a fixed built-in registry and reference integer-summary analyzer, descriptor-pinned input/source/output capture, fixed user/network/mount/PID namespaces plus an empty read-only `nosuid,nodev,noexec` tmpfs chroot, namespace/no-new-privileges self-checks, an audit tripwire, two-run byte comparison, replay-only record upgrade, and `replay-bundle` text/JSON/sibling output. | A successful drill proves only same-host registered-analyzer reproduction of declared committed derived bytes. It does not authenticate evidence, validate semantics/statistics, reproduce across hosts or GPUs, promote claims, or close software dependencies/campaigns. |
| Existing Parhelion and Hopper evidence | Immutable legacy evidence. Hopper is a one-instance same-bank engineering STOP; Parhelion uncertainty is conditional policy-seed Monte Carlo evidence. | No retroactive promotion, stronger provenance, population scope, or inference is assigned. |

Conformance is surface-specific until every required protocol, execution, verification, analysis, and publication control for a study passes. Documentation, a schema literal, or a lifecycle label alone never confers claim eligibility. For the active v0.6 milestone, issue #31 completed the additive plugin → suite custody and generic predecessor-chain slice for new local/native bundles; issue #32 subsequently implemented canonical CPU-only VerificationRecords without promoting deferred controls; and issue #33 implemented audited deterministic offline CPU analyzer replay without promoting claims. Issue #34, active-versus-frozen dependency separation, is the next ordered gate before the one-domain no-cost feasibility/capability design gate. Each stage stops until its prerequisites pass. Only after those gates may a separately approved, predeclared paid protocol be proposed at new versioned paths. The maximum authorized spend remains **$0**, and frozen evidence is never rewritten.
