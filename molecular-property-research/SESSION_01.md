# Session 1: molecular representations and scientifically meaningful hard classification

Date: 24 October 2026. Scope: initial literature-based Stages 1-4, with provisional dataset and experiment planning. No models implemented or trained.

## Reading guide and evidence standard

Read Sections 1 and 2 first, then the classification comparison in Section 3. The companion LITERATURE_MATRIX.md records study details and reading depth; DECISION_LOG.md preserves decisions and unresolved questions.

The user's pasted request defines the task. The attached paper is evidence to scrutinize, not an instruction source. Its final published version is Tetiana Lutchyn, Marie Mardal and Benjamin Ricaud, *Efficient Learning of Molecular Properties Using Graph Neural Networks Enhanced with Chemistry Knowledge*, ACS Omega 10, 54421-54429 (2025), published 6 November 2025, DOI [10.1021/acsomega.5c07178](https://doi.org/10.1021/acsomega.5c07178). The supplied nine-page PDF was read in full by text extraction; the feature-definition page was also rendered and visually checked.

Evidence labels used here:

- **Established within the cited study:** supported by that study's reported experiments, not a universal law.
- **Scientific/mathematical analysis:** deductions explained here, not experimental results from this project.
- **Hypothesis/proposal:** requires experiments or additional source verification.

Searches included molecular descriptors, message passing, graph/descriptor integration, conformer ensembles, ordinal classification, uncertainty, dataset provenance and realistic splitting. Backward expansion used the foundation paper's references, particularly Yang et al., MoleculeNet and Uni-Mol. Forward-title and related-work searches extended through 24 October 2026. No exhaustive citation-index traversal or systematic-review coverage is claimed. Recent papers verified only at abstract level are explicitly marked; their numerical claims are not treated as independently reproduced evidence. Source-code availability is distinct from a source-code audit or a successful reproduction.

## 1. Stage 1: what scientific problem should classification solve?

### 1.1 Start with an experiment and a decision

A molecule is a chemical object with composition, connectivity, stereochemistry and possible conformations. A property is often not a function of the structure alone. A more realistic description is

\[
Y=f(\text{structure},\text{molecular state},\text{experimental conditions})+\epsilon.
\]

State includes protonation, tautomer and conformer; conditions may include pH, temperature, solvent, dose, assay protocol and biological target. Experimental noise and source disagreement enter through epsilon. For example, lipophilicity expressed as logD at a specified pH is different from logP of the neutral species. Molecular target inhibition is different from clinical benefit, and assay toxicity is different from organism-level safety.

In drug discovery, ADMET means absorption, distribution, metabolism, excretion and toxicity. These endpoints support different decisions: which compound to assay next, which liability to investigate, whether a candidate can reach an assay concentration, or which chemical series to optimize. [Therapeutics Data Commons](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/4c56ff4ce4aaf9573aa5dff913df997a-Abstract-round1.html) organizes tasks across this pipeline; this does not make every endpoint equally suitable for our study.

**Scientific purpose of a categorized dataset:** give each compound an operational label with a documented meaning, so researchers can filter, compare groups, allocate experiments and apply a consistent decision rule. A useful dataset retains the measured value when available, units, assay context, label provenance, category rule and uncertainty alongside its primary categorical output.

**What classification adds over continuous predictions:** an explicit link to an action, evaluation of asymmetric mistakes at the action boundary, and a consistent interface for screening. These benefits are not exclusive to a directly trained classifier. A sufficiently informative regression distribution plus a decision rule can provide them too. Hard labels alone contain less information than the measurements from which they were derived.

### 1.2 Problem map

| Scientific setting | Question | Possible categorical output | What must accompany the label |
|---|---|---|---|
| Target screening | Which compounds deserve confirmatory testing against target T? | Meets/does not meet a specified assay activity criterion | Target, assay, potency units, concentration, interference flags |
| ADMET liability | Which compounds should receive early follow-up liability testing? | Assay liability positive/negative | Endpoint-specific definition and false-negative cost |
| Solubility feasibility | Can this compound plausibly support the planned assay concentration? | Below/above the required solubility; possibly justified ordered bands | pH, temperature, equilibrium/kinetic measurement, solvent and required concentration |
| Lead optimization | Which modifications move a series into a desired property window? | Below/in/above a window | Window justification; continuous values retained for within-class ranking |
| Multiple biological assays | Which pathways respond? | Several independent binary labels per compound | Missing labels and assay-specific definitions |

### 1.3 Three different kinds of category

**Operationally categorical endpoints:** a protocol records an outcome such as an Ames positive/negative call. These are categorical at the reporting level, even though the underlying biology is not inherently binary. Validity depends on protocol and how conflicting tests were combined. Do not interpret Ames negative as globally nontoxic.

**Scientifically justified discretization:** a continuous endpoint is split because a particular workflow needs a threshold. The hERG_Karim benchmark, for example, uses inhibition potency below 10 micromolar versus at least 10 micromolar. That is the dataset's operational definition, not a universal guarantee of cardiac safety. [TDC toxicity documentation](https://tdcommons.ai/single_pred_tasks/tox/)

**Statistical discretization:** median, quantile or equal-width bins can make a useful methodological test but do not establish biological classes. Quantiles depend on which compounds were sampled. A compound can change class when the dataset changes without any chemistry changing. If training-derived quantiles are used in a secondary experiment, fit them on training data only and freeze them; call them statistical categories.

Thresholds require two separate choices:

1. A **measurement threshold** tau defines the true endpoint, for example C=1[Y<tau] for a lower-is-more-active potency measurement.
2. A **prediction decision threshold** t converts predicted probability p into a hard call, C_hat=1[p>=t]. It need not equal 0.5.

Confusing these thresholds makes experiments uninterpretable. For calibrated p and zero costs for correct decisions, the loss-minimizing positive call is p>C_FP/(C_FP+C_FN), where C_FP and C_FN are the costs of false positives and false negatives. This is a mathematical decision rule, not a proposed clinical policy.

### 1.4 Three promising applications

| Application | Proposed scientist benefit | Why promising | Principal limitation |
|---|---|---|---|
| hERG inhibition liability | Prioritize confirmatory electrophysiology or early liability review | Concrete endpoint and established binary data; investigate recall versus follow-up burden | Heterogeneous assays, threshold noise, exposure and mechanism absent from structure-only labels |
| Ames mutagenicity | Prioritize bacterial mutagenicity testing or chemical-series review | Operational categorical labels; plausible role for local structural motifs | Strain/metabolic activation and aggregation rules matter; not a universal human toxicity label |
| Solubility for assay feasibility | Avoid testing compounds at concentrations they cannot sustain under specified conditions | Closely connects to TChemGNN and permits direct classification versus thresholded regression | Public measurements may lack the conditions required to defend the category |

For solubility, logS=-4 in mol/L means 100 micromolar; -5 means 10 micromolar. These are unit conversions, **not adopted category boundaries**. A threshold requires an actual assay concentration, acceptable margin and compatible measurement conditions. Drug formulation solubility also depends on dose and dissolution context; there is no universal high/medium/low solubility partition justified here.

### 1.5 Stage 1 checkpoint

1. **Investigated:** screening, ADMET, assay calls, property windows and categorization.
2. **Literature establishes:** public molecular tasks include both measured continuous endpoints and assay-derived categories; definitions vary by benchmark.
3. **Uncertain:** our scientist's actual action, acceptable errors and compatible experimental conditions.
4. **Competing approaches:** categorical assay prediction, decision-based discretization, regression plus an action rule, or ranking at a fixed testing budget.
5. **Opportunity:** connect evaluation to a defensible experimental decision instead of arbitrary bins.
6. **Next step:** draft task cards for the three applications above and audit label provenance.
7. **Your study task:** explain why “hERG blocker,” “toxic,” and “unsafe drug” are different statements.
8. **Advance when:** one endpoint has a documented label rule, named user/action and usable data; a benchmark score alone is insufficient.

## 2. Stage 2: representations from first principles and the 35-feature audit

### 2.1 What a representation must preserve

SMILES is a line notation for molecular connectivity, with optional stereochemical and isotope information. It is not normally a measured 3D structure. Multiple valid SMILES can describe the same molecular graph. Canonicalization selects one string under a particular algorithm. [Daylight arbitrary SMILES documentation](https://daylight.com/dayhtml/doc/man/man3/dt_arbsmiles.html)

A molecular graph G=(V,E) represents atoms as nodes and bonds as edges. Node features describe atom environments; edge features can encode bond order, aromaticity, conjugation and stereochemistry. A descriptor is an explicitly computed quantity such as molecular weight. A fingerprint encodes structural patterns as bits or counts. A conformer is one spatial arrangement compatible with the molecular structure.

A useful representation should distinguish chemistry relevant to the endpoint while ignoring irrelevant coordinate frames and atom numbering. Molecule-level scalar predictions should ordinarily be invariant under atom relabeling, translation and rigid rotation. Chirality needs separate treatment: reflection invariance is inappropriate if the endpoint distinguishes enantiomers. A representation should retain protonation and stereochemistry when relevant, handle molecules outside its training vocabulary, and permit detection of unsupported inputs.

Crucial limitation: if two molecules or molecular states map to exactly the same representation but have different target distributions, the downstream model cannot recover the missing distinction from that representation alone.

### 2.2 What TChemGNN actually does

The paper defines, for atom i,

\[
x_i=[a_i\in\mathbb R^{14};\ d(G)\in\mathbb R^{15};\ g(G,R)\in\mathbb R^6].
\]

For N atoms, the input is N by 35. The same 21 global values are repeated at every atom. This is not a single 35-dimensional vector per molecule, and it is not 35 independent chemical factors. The model uses graph attention layers, omits explicit edge attributes, and explores pooled versus selected-node outputs. Its tasks are **regression**, including the quantitative BACE endpoint. Its reported regression results do not establish classification benefits. See supplied PDF, pp. 54422-54427.

The authors' main useful proposition is that supplying inexpensive chemical summaries can help a small graph model. The idea should be tested independently of their readout and feature choices. Graph/descriptor integration predates this paper: [Yang et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6727618/) reported improvements on some tasks and deterioration on others. “Add all descriptors” is therefore not an evidence-based universal recommendation.

### 2.3 All 35 features

The inventory and scaling below follow the supplied PDF's feature-definition page, p. 54424. Calculation descriptions are **an operational interpretation to verify against implementation**, not a claim that the paper uniquely specifies every API call. The public repository was located, but the preprocessing source could not be retrieved in this session. H denotes hydrogen. “Potential use” is chemical rationale, not an observed feature-specific improvement. Every atom-level feature is local; every molecular/geometry entry is global even though the model repeats it locally.

#### Fourteen atom-level entries

| # / feature | Chemical meaning and calculation | Potential use | Redundancy, limitations and data sensitivity |
|---|---|---|---|
| 1 atom_degree | Number of directly bonded graph neighbors; distinguish neighbor count from bond-order sum | Branching, substitution, local motifs for activity/toxicity | Explicit H changes degree; correlated with H count and valence; the paper's phrase “number of bonds” needs an exact convention |
| 2 atomic_number | Element identity Z | All endpoints; heteroatom and element composition | Does not encode charge or environment; raw numeric Z imposes a numerical ordering a learned encoding need not assume |
| 3 num_hydrogens | Attached H count, normally explicit plus implicit under a chosen convention | Donor groups, saturation and local reactivity | Protonation/tautomer sensitive; correlated with valence and degree |
| 4 atomic_valence | Valence under chosen explicit/total and bond-order conventions | Bonding environment and reactive motifs | Not oxidation state; aromaticity and unusual valence handling matter; code verification required |
| 5 num_radical_electrons | Number of unpaired electrons assigned by graph sanitization | Radical-containing chemistry | Often constant zero in drug-like sets; incorrect structure parsing changes it |
| 6 atom_formal_charge | Integer formal charge in the represented Lewis structure | Ionization-related solubility, permeability and binding | Formal charge is not partial charge; protonation and salts matter |
| 7 atom_hybridization | Assigned sp/sp2/sp3 etc. from bonding rules | Planarity, conjugation and local shape | Categorical; encoding enum values as continuous can impose spurious distances; toolkit perception dependent |
| 8 electronegativity | Element-table lookup under a declared scale | Polarity/charge tendency, possibly solvation | Determined by element identity once table fixed; does not encode environment-dependent charge |
| 9 has_electronegativity | Indicator that lookup exists | Missing-value handling | Mostly constant for common elements; can encode preprocessing artifacts; define fallback separately |
| 10 has_implicit_hydrogens | Boolean indicating an implicit H is present | Representation bookkeeping | Can change with explicit-H conversion without chemistry changing; audit necessity |
| 11 hydroxyl_group | Identify atoms participating in O-H groups through a declared graph/substructure rule | Solubility, hydrogen bonding, metabolic motifs | Paper's wording is insufficient to specify whether O, H or multiple atoms are flagged; not all donors are hydroxyls |
| 12 scaled atomic mass | (A-10.812)/116.092 | Composition and size-related effects | Mostly element-determined, isotope-dependent; constants are not dataset standardization or guaranteed bounds |
| 13 scaled van der Waals radius | (R_vdw-1.5)/0.6 from an element table | Steric size proxies | Mostly redundant with Z; radius convention matters; not a measured per-atom volume |
| 14 scaled covalent radius | (R_cov-0.64)/0.76 from an element table | Typical bond-length/size tendencies | Element-table proxy, not actual bond length; table and units must be recorded |

#### Fifteen molecular-level entries

| # / feature | Chemical meaning and calculation | Potential use | Redundancy, limitations and data sensitivity |
|---|---|---|---|
| 15 has_ring | Boolean for any perceived ring | Rigidity, scaffold and activity | Usually derivable from ring count; ring perception convention matters |
| 16 is_aromatic | Molecule-level aggregation of aromatic atom/bond flags | Conjugation, stacking and hydrophobic motifs | “Any aromatic atom” versus other aggregation is unspecified; loses location and number |
| 17 formal_charge | Sum of atom formal charges | Ionic solubility, permeability and binding | Sum of entry 6; neutral total can conceal separated positive/negative sites |
| 18 min_degree | Minimum graph atom degree | Coarse topology | Often nearly constant in connected organic molecules; explicit H/disconnected ions alter it |
| 19 num_hbond_donors | Count donor sites using a specified substructure definition | Solvation, permeability and binding | Apparently overlaps entry 26; chemical donor rules and protonation matter |
| 20 num_rings | Count perceived rings | Rigidity and scaffold complexity | Cycle bases are not unique; related to has_ring but richer |
| 21 num_rotatable_bonds | Count rotatable bonds under a declared rule | Flexibility, entropy and conformer diversity | Amide, terminal and ring exclusions differ by definition; no conformer populations encoded |
| 22 polar_surface_area | Topological polar surface area from atom-fragment contributions | Solubility, membrane permeability | A 2D estimate, not explicit 3D solvent-accessible area; correlated with donors/acceptors |
| 23 molecular_weight | Sum composition-based atomic weights, with declared isotope convention | Size and physicochemical trends | Correlated with atom count, mass and volume; not sufficient to distinguish isomers |
| 24 num_atoms | Graph atom count or declared total count | Molecular size | Explicit versus implicit H changes the value; heavy-atom count must be distinguished |
| 25 hba | Hydrogen-bond acceptor count using a declared rule | Solvation and binding | Depends on protonation, resonance and rule definitions; location lost |
| 26 hbd | Hydrogen-bond donor count | Same intended role as entry 19 | Treat as a suspected duplicate, not proven duplicate until API definitions/output equality checked |
| 27 fraction_sp2 | Number of sp2 atoms divided by declared atom population | Planarity and conjugation | Denominator unclear; not RDKit FractionCSP3 and not simply 1-FractionCSP3 |
| 28 valence | Sum of chosen atom valences | Coarse composition/bonding | Aggregation of entry 4; strongly size-related; sum discards distribution |
| 29 general_electronegativity | Mean element electronegativity over a declared atom set | Coarse compositional polarity tendency | Mean of entry 8, missing handling and H convention matter; positional information lost |

#### Six geometry-related entries

| # / feature | Chemical meaning and calculation | Potential use | Redundancy, limitations and data sensitivity |
|---|---|---|---|
| 30 volume | Estimate conformer molecular volume; paper identifies ComputeMolVolume | Steric size and solvation | Grid, H, radii and conformer dependent; not simply width*height*length; correlated with size |
| 31 width | Coordinate extent along one Cartesian axis: max(r_axis)-min(r_axis) | Rough shape if a meaningful frame is fixed | Changes under rotation of the same conformer; no standardized physical frame specified |
| 32 length | Extent along another Cartesian axis | Same potential shape role | Same rotation issue; sorted axis extents still need not be rotation invariant |
| 33 height | Extent along remaining Cartesian axis | Same potential shape role | Same rotation issue; a single conformer can be unrepresentative |
| 34 dipole momentum | Approximate dipole from partial charges and coordinates; paper mentions MMFF/Gasteiger | Electrostatics and solvation | Exact method/units unspecified; Gasteiger and MMFF charges are different approximations; magnitude is rotation invariant, components are not; charged-molecule dipole depends on origin |
| 35 angle of general molecular orientation | Some orientation angle derived from an optimized conformer; paper mentions UFF | No intrinsic isolated-molecule scalar meaning established | Axis/reference/formula unspecified; absolute orientation is arbitrary; principal-axis degeneracy can destabilize angles |

Aromaticity and valence assignments are toolkit conventions; [RDKit's own book](https://www.rdkit.org/docs/RDKit_Book.html) documents its perception rules. Record toolkit version, H policy, protonation, salt handling, tautomer treatment and stereochemistry before comparing features.

### 2.4 Scientific critique of the feature and model claims

**Redundancy need not mean uselessness.** Element-derived mass, radii and electronegativity can make useful priors easier for a small model to exploit, even when they add no new information beyond Z. Deterministic global descriptors can similarly reduce the learning burden without adding information absent from a fully specified graph. Remove suspected duplicates in a controlled ablation; do not infer causal importance from correlated feature importances.

**The geometry is generated, not observed.** A single embedded, force-field-optimized conformer is not guaranteed to be the dominant aqueous or protein-bound state. A solubility measurement also depends on crystal packing and conditions absent from a free-molecule conformer. [MARCEL (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7eb6233e02f7d9efbb84acd839a996fb-Abstract-Conference.html) explicitly studies conformer ensembles; using multiple conformers is an existing research direction, not a new idea by itself.

**Orientation is a testable concern.** A rigidly rotated isolated molecule should have the same scalar endpoint. The paper's axis extents and undefined orientation angle do not automatically satisfy this. A later preprocessing audit should rotate a fixed conformer and compare features/predictions. Candidate alternatives include principal moment ratios or pairwise-distance summaries; their usefulness still requires comparison. RDKit provides [3D shape descriptors](https://rdkit.org/docs/source/rdkit.Chem.Descriptors3D.html). A principal-axis frame can have sign and degeneracy ambiguities; alignment alone is not a universal fix.

**Dipole requires precision.** An approximate point-charge construction is mu=sum_i q_i(r_i-r0), with magnitude ||mu||. For total charge Q, changing origin by a gives mu'=mu-Qa. Thus charged molecules require a declared origin; approximate charges and geometry do not produce an experimentally validated dipole automatically.

**Missing information is substantial.** The feature list lacks explicit bond-order/stereo edge channels, explicit chiral-center descriptors, environmental metadata, protein context, pKa and conformer populations. This does not imply each is necessary for every endpoint. It means 35 is a candidate design, not a chemically complete representation.

**Selected SMILES nodes are not chemical explanations.** The paper asserts that first/last SMILES atoms are peripheral in a chemically useful way. Arbitrary valid SMILES need not preserve this identity; canonicalization is algorithm-dependent. A cycle such as benzene has no degree-one terminal atom, despite having a first and last textual atom. Global descriptors are also repeated at every node, so a node prediction can use molecule-wide information. Prediction accuracy at one node does not establish that its atom causes the property. Randomized-SMILES and atom-relabeling controls are essential if this claim becomes part of our study.

**Performance claims need matched protocols.** Default splits in the paper are random 80/10/10; some comparisons report cross-validation. Table 3 labels a FreeSolv result as validation, while Table 5 discusses test results; split identities need auditing. Table 5 also excludes molecules with fewer than five atoms for that analysis. Published scores from different splits and filtering rules cannot prove model superiority. Table 2's ESOL max-pooling RMSE (0.7752) is slightly below its selected-node model (0.7844), so the evidence does not establish universal last-node superiority even within those reported rows. The discussion's treatment of lipophilicity literature is not a basis for claiming no prior regression research. These observations motivate verification, not an allegation of misconduct.

**Geometry failure can bias evaluation.** The paper excludes two lipophilicity molecules after preprocessing failures; its main text and footnote describe the failure differently. Retained cohort, exclusions and failure handling must be matched across baselines. Failed conformer generation is not evidence that a molecule should be discarded scientifically.

### 2.5 Independent comparison of representation families

The following is a design comparison, not a benchmark ranking. [Rogers and Hahn's ECFP paper](https://doi.org/10.1021/ci100050t), [Gilmer et al.'s MPNN framework](https://proceedings.mlr.press/v70/gilmer17a) and [Uni-Mol](https://openreview.net/pdf?id=6K2RM6wVqKu) are representative primary sources.

| Representation | Preserves/learns | Strength | Main limitation | Relative cost and classification role |
|---|---|---|---|---|
| Small 2D descriptor vector | Explicit composition, polarity and topology summaries | Fast, interpretable baseline; works with linear/tree models | Can collapse distinct structures; descriptor definitions matter | Low preprocessing/training cost; essential baseline |
| Morgan/ECFP fingerprint | Iteratively encoded local neighborhoods, hashed as bits/counts | Strong structural similarity and classical ML baseline | Hash collisions, finite radius and stereo settings; limited explicit geometry | Low feature cost; compare linear and nonlinear classifiers |
| Graph kernels | Similarity from walks/substructures | Useful structured-data alternative | Large pairwise kernel matrices may scale poorly | Modest datasets; a worthwhile optional control |
| Atom/bond GNN or D-MPNN | Learned neighborhoods and molecular readout | Learns task-adapted structural patterns | Limited receptive field/expressivity; data and tuning needs | Moderate cost; use standard implementation before inventing architecture |
| Graph plus global descriptors | Learned structure plus supplied chemistry summaries | Can reduce burden of learning global statistics | Redundancy and task-specific harm; fusion confounds architecture/feature effects | Moderate; compare input broadcast with post-readout concatenation |
| SMILES/SELFIES sequence model | Serialized molecular tokens, often pretrained | Transfers patterns from unlabeled corpora | Serialization dependence; pretraining overlap and resource costs | Fine-tuning moderate to high; pretraining high; later comparator |
| 3D invariant/equivariant model | Distances, directions, angular structure | Models spatial relationships explicitly | Geometry quality, conformer coverage and stereochemical design | Higher feature/training cost; conditional on credible geometry need |
| Conformer ensemble representation | Distribution of shapes or predictions over conformers | Can account for molecular flexibility | Ensemble generation and weighting may be inaccurate or expensive | Higher, approximately grows with conformer count; test only after 2D controls |
| Multimodal/pretrained representations | Several structure/property/text modalities | Potential transfer and complementary information | Data overlap, missing modalities, expensive tuning and attribution of gains | Highest integration burden; unnecessary for first experiments |

Fingerprints and descriptors are representations, not classifiers. The same fingerprint can feed logistic regression, a random forest, a neural classifier or a regressor. Likewise, a GNN encoder can support either classification or regression. Hard classification does not intrinsically demand a special representation family.

### 2.6 Stage 2 checkpoint

1. **Investigated:** SMILES, graphs, descriptors, fingerprints, geometry and all 35 entries.
2. **Literature establishes:** chemical descriptors can complement learned graphs, with task-dependent effects; the foundation paper reports regression evidence under its protocols.
3. **Uncertain:** exact implementations of several features, duplicate donor definitions, geometry conventions and generalization to classification.
4. **Competing approaches:** compact 2D descriptors, fingerprints, bond-aware graphs, hybrid representations and credible 3D inputs.
5. **Opportunity:** determine which feature groups help a specified endpoint and whether gains survive chemistry-preserving perturbations and realistic splits.
6. **Next step:** build a feature specification sheet before any training; audit public preprocessing against the published definitions.
7. **Your study task:** manually compare ethanol, dimethyl ether and acetic acid: connectivity, donors/acceptors, charge state and what a size-only vector misses.
8. **Advance when:** every chosen input has a reproducible definition, unit, missing-value policy and invariance expectation, and simple baseline representations are specified.

## 3. Stage 3: how should hard classification be formulated?

### 3.1 Taxonomy

| Formulation | Output/training | Appropriate use | Critical comparison |
|---|---|---|---|
| Binary | Probability p, binary cross-entropy or classical margin/tree objective; hard call at t | One operational yes/no endpoint | Threshold and error costs must be explicit; AUROC alone does not evaluate deployed calls |
| Nominal multiclass | K probabilities, multiclass cross-entropy; argmax or cost-based action | Mutually exclusive unordered outcomes | Treats all wrong classes symmetrically unless a cost matrix changes the decision/loss |
| Multilabel | One probability per endpoint; masked binary losses | Multiple assays can be positive simultaneously | Missing is not negative; endpoint prevalences and assay sizes differ |
| Ordinal classification | Ordered-logit/probit or cumulative-threshold models | Low/intermediate/high only when scientifically defensible | Uses class order without requiring equal numeric spacing; thresholds must be consistent |
| Point regression then categorization | Train on measured Y with squared/absolute/robust loss, then apply fixed measurement boundaries | Continuous measurements available | Preserves within-class training information; low RMSE does not guarantee low boundary error |
| Distributional regression then categorization | Estimate F(Y|x), integrate probability over category intervals | Need category probabilities and quantitative uncertainty | Requires defensible distribution and calibration; stronger comparator than only thresholding a point estimate |
| Joint regression/classification | Shared representation with weighted losses | Same measurements support both tasks or related assays | Loss weighting and tuning add confounding; derived category is not independent supervision |
| Cost-sensitive/selective decision | Probability model plus expected-loss decision and optional review flag | Screening with asymmetric costs or limited follow-up budget | Report automated coverage and review burden; do not present abstention as a chemical class |

For binary cross-entropy, L=-[c log p+(1-c)log(1-p)]. It rewards probability assigned to the observed class; it does not by itself choose the operational hard-call threshold. Class weights and resampling can improve minority-class attention but alter probability interpretation and require calibration checks. Focal loss is a candidate for difficult examples, not a default: mislabeled boundary points may receive extra influence.

For K ordered classes, cumulative models estimate q_k=P(C>k|x), k=0,...,K-2, constrained so q_0>=q_1>=... . Recover class probabilities by differences between successive cumulative probabilities, with the outer endpoints 1 and 0. [CORAL, Cao, Mirjalili and Raschka (2020)](https://sebastianraschka.com/pdf/papers/2020-coral.pdf) is a rank-consistent neural example validated on age estimation; that paper does not demonstrate molecular superiority. Begin with ordinary ordered logistic regression before a specialized neural head if ordered categories are chosen.

### 3.2 Why direct classification and thresholded regression can disagree

For category C=1[Y<tau], the Bayes-optimal binary prediction under equal misclassification costs depends on P(Y<tau|x), while squared-error regression estimates E[Y|x]. Thresholding the mean need not yield the most likely category.

Illustrative construction, not molecular data: if Y=-2 with probability 0.6 and Y=4 with probability 0.4 and tau=0, most measurements are negative but E[Y]=0.4 is positive. A perfectly estimated mean thresholded at zero chooses the less likely class. This does not prove direct classification wins in finite data: regression also learns distances and within-class structure that discretization discards.

If a regressor estimates a conditional CDF F, ordered category probabilities follow

\[
P(C=k|x)=F(b_{k+1}|x)-F(b_k|x),
\]

with a consistent boundary convention, infinite outer bounds, and adjustments for any point masses. Under a continuous Gaussian predictive distribution, a lower-than-tau event has probability Phi((tau-mu)/sigma). This is a proposed comparator only if its distributional assumptions withstand checks.

### 3.3 Information loss and boundary uncertainty

Discretization merges different measurements inside a class and makes almost identical measurements on opposite sides appear categorically different. A hard class cannot support fine within-class potency ranking. Keeping original measurements enables re-thresholding for a new scientist's workflow. More bins can increase apparent detail while reducing per-class sample size and making noisy boundaries dominate.

For a true continuous value y and Gaussian measurement error with standard deviation sigma, the probability that a single measured value lands on the wrong side of tau is Phi(-|y-tau|/sigma). It approaches one half at the boundary. This is an illustrative noise model; actual replicate variability must be estimated, not assumed Gaussian. Boundary uncertainty, activity cliffs, rare classes and unfamiliar scaffolds are different sources of failure and should be analyzed separately.

Censored values such as IC50>10 micromolar can establish some classes unambiguously but cannot be replaced by an exact value of 10 for regression. Retain inequality flags. Conflicting repeats should be investigated or modeled; silently averaging incompatible assays can create meaningless labels.

### 3.4 Reliability, imbalance and metrics

Retain probability estimates even though the primary delivered label is discrete. Calibration asks whether events assigned probability 0.8 occur about 80% of the time in the population being evaluated. Accuracy, ranking and calibration are different. Temperature scaling is an established postprocessing baseline from [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html); its evidence is not a guarantee under chemical distribution shift.

Report balanced accuracy=(sensitivity+specificity)/2 for binary tasks, macro-F1, per-class precision/recall, confusion matrices and counts. Include AUROC and average precision (state the exact AUPRC estimator), with positive class and test prevalence stated. Precision depends on prevalence. For ordered categories add mean absolute class-distance error and severe-error frequency; equal class-distance costs are an analysis choice, not automatically scientific costs. For calibration use Brier score, log loss and reliability diagrams; ECE is bin-sensitive and inadequate alone.

Evaluate scientific utility as, for example, recovered assay-positive compounds at a fixed follow-up budget, or specificity at a predeclared sensitivity target. Choose the target with the intended researcher; no arbitrary target is adopted here. Compare with random selection, prevalence-only predictions and simple physicochemical rules where justified.

Conformal prediction can return sets rather than a single class. Under exchangeability it offers marginal coverage, not guaranteed correctness for every molecule or unfamiliar scaffold. A set spanning classes can trigger review while the primary prediction remains a hard label. [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511) explain the framework. Do not claim ordinary conformal validity under arbitrary scaffold shift. Distinguish uncertain measurements from lack of model knowledge; ensembles need not cleanly identify either component.

### 3.5 Research representation and prediction separately, then together

Use a crossed design rather than changing everything at once:

| Question | Hold fixed | Vary | Permitted conclusion |
|---|---|---|---|
| Representation effect | Labels, splits, model family, tuning budget and decision policy | Fingerprints, descriptors, their combination | Which representation works better for that model and task |
| Prediction formulation effect | Representation, compounds/splits and comparable tuning | Direct classifier versus point/distributional regression; ordinal only if applicable | Effect of learning objective/decision construction in this setting |
| Graph feature-group effect | Backbone, bond inputs, readout and training budget | Local-only, plus 2D globals, plus geometry | Incremental benefit conditional on this graph pipeline |
| Fusion effect | Exactly the same features and comparable capacity | Input broadcast versus late concatenation | Whether integration placement matters |
| Interaction | Shared task, splits and metrics | Cross promising representations with objectives | Whether one representation's advantage depends on the objective |

This controls avoidable confounding without pretending fingerprint and graph architectures can be identical. Analyze paired score differences on the same evaluation compounds. Frozen embeddings permit a clean head comparison but answer a different question from fully retrained task-specific encoders; label the experiment accordingly.

### 3.6 Stage 3 checkpoint

1. **Investigated:** binary, nominal, multilabel, ordinal, regression-derived and joint formulations; uncertainty and decisions.
2. **Literature/mathematics establishes:** ordered labels admit order-aware models; thresholded means and optimal category decisions differ in general; calibration requires separate evaluation.
3. **Uncertain:** best formulation for our endpoint, available continuous measurements and boundary noise.
4. **Competing approaches:** direct classification, distributional regression, ordinal models and simpler cost-based decision rules.
5. **Opportunity:** compare meaningful hard decisions rather than comparing unrelated regression and classification scores.
6. **Next step:** specify measurement boundaries, decision policy and one primary utility metric before training.
7. **Your study task:** work through the two-point conditional-distribution example above and explain why RMSE and classification accuracy can disagree.
8. **Advance when:** competing methods can be scored against identical categorical labels and decisions without leakage.

## 4. Stage 4: literature trends, limitations and research opportunities

The companion matrix records 15 core studies/resources plus recent leads. Main conclusions:

- **No universal representation winner follows from this evidence.** Chemistry summaries, fingerprints and learned graphs capture different biases; task, sample size and evaluation protocol matter. Yang et al. already provide direct evidence against assuming descriptor additions always help.
- **Geometry is conditional on input quality and target.** Spatial models and conformer ensembles are established approaches. The open question is whether their additional cost and uncertainty buy better decisions on our selected endpoint.
- **Reliability and provenance can dominate architecture.** The original MoleculeNet study and TDC define useful benchmarks but their prepared datasets are not universal scientific ground truth. A recent [Tox21 leaderboard preprint](https://arxiv.org/abs/2511.14744) reports dataset/label changes across integrations; verify a particular release before attributing results to algorithms.
- **Unfamiliar scaffold is only one kind of unfamiliar chemistry.** A 2025 [NCI-60 splitting study](https://link.springer.com/article/10.1186/s13321-025-01039-8) reports substantial protocol dependence. A [2026 structural-frontier preprint, v4 dated 6 October](https://arxiv.org/abs/2607.10729v4) proposes additional stress testing. Neither establishes that one split universally predicts prospective performance. Time/source holdouts are valuable when available.
- **Difficult local cases deserve separate analysis.** [MoleculeACE](https://doi.org/10.1021/acs.jcim.2c01073) studies activity cliffs; ordinary overall averages can hide poor performance there. Chemical activity cliffs are not the same as arbitrary discretization boundaries. Its published correction must be checked before reproducing results.
- **Recent uncertainty work narrows novelty.** [Lee et al. (August 2026)](https://arxiv.org/html/2608.17678v1) study solubility uncertainty under label shift using a regression model. It is relevant prior work, not proof of reliable hard classification under arbitrary chemical shifts.

### 4.1 Ranked provisional gaps and falsifiable hypotheses

These are project directions, **not verified unaddressed gaps**. Ranking favors information gained per undergraduate-sized experiment. Each needs a focused novelty search after choosing an endpoint.

| Rank / hypothesis | Rationale and prior evidence | Smallest credible later experiment | Support / rejection | Feasibility, novelty and risk |
|---|---|---|---|---|
| 1. A compact audited 2D feature subset improves hard decisions across chemical splits | TChemGNN motivates globals; Yang shows task-dependent effects | Fix a standard bond-aware graph; local-only versus +15 global 2D versus duplicate-pruned subset; same splits/tuning | Support: paired improvement in predeclared utility across splits without calibration harm; reject meaningful improvement if uncertainty rules it out | Moderate compute; generic descriptor addition has low novelty; carefully established endpoint-specific robustness may be useful |
| 2. Training formulation changes boundary reliability more than overall AUROC suggests | Category decisions depend on conditional mass, not only mean | On one continuous endpoint with a defensible threshold, fix fingerprints; compare direct classification, point regression and a checked distributional regressor | Support: reliable differences in boundary error/utility on held-out compounds; reject broad superiority if benefits depend on a favorable split or boundary | Low/moderate compute; novelty uncertain; biggest risk is inadequate threshold/measurement provenance |
| 3. Geometry improvements fail unless invariance and conformer uncertainty are controlled | Paper's axis extents/angle motivate audit; MARCEL and 2026 work already study ensembles | Fixed 2D baseline plus paper-like versus invariant 3D summaries; repeated rotations and conformer seeds, common cohort | Support: invariant features preserve predictions and retain useful gains; reject utility claim if no improvement over 2D after cost/failure accounting | Moderate; strong audit question, but contribution needs more than demonstrating rotation dependence by construction |
| 4. Descriptor gains disappear or reverse on chemically distant compounds | Existing split studies motivate conditional generalization | Compare paired gains on scaffold and similarity/source holdouts; include train-neighbor similarity analysis | Support: reproducible interaction between representation and evaluation domain; reject if gains remain stable within useful uncertainty bounds | Moderate; evaluation work is active, so avoid claiming new split novelty prematurely |
| 5. Calibration/review policy improves screening utility without improving ranking | Calibration literature motivates probability-aware decisions | Freeze one classifier, calibrate on held-out data, compare fixed-threshold and cost-aware/review policies | Support: lower error burden at declared coverage/budget on external holdout; reject if benefit vanishes under shift or costs | Low compute; existing general methods, potential value in endpoint-specific validation |

Alternative explanations to investigate: more trainable parameters, unequal tuning, duplicate descriptors, molecule-size shortcuts, dataset-source identity, class-prevalence shifts, pretraining overlap and selectively discarded geometry failures. A negative result that rules out a plausible improvement under credible evaluation is informative.

### 4.2 Stage 4 checkpoint

1. **Investigated:** foundational representations, descriptor integration, geometry, benchmarks, reliability and related work through the session date.
2. **Literature establishes:** these broad methodological ideas already exist; benchmark superiority depends on protocol and is not a scientific contribution by itself.
3. **Uncertain:** novelty of any narrow endpoint-specific hypothesis; several recent leads have only abstract-level verification.
4. **Competing explanations:** representation quality versus data quality, split difficulty, tuning, boundary design or calibration.
5. **Opportunity:** a controlled, decision-relevant reliability study with a compact representation and transparent negative results.
6. **Next step:** deeply read the three or four nearest papers for the chosen hypothesis, including corrections and supplements; complete forward-citation review.
7. **Your study task:** fill one literature-matrix row yourself and identify which claim its experiments cannot support.
8. **Advance when:** we can state a precise unresolved question, distinguish it from nearest prior work and define what would falsify our answer.

## 5. Candidate datasets: provisional shortlist, not completed Stage 5

Counts below describe cited releases before our own parsing/curation. They are not claims about final usable rows or class balance. Official sources: [TDC toxicity](https://tdcommons.ai/single_pred_tasks/tox/), [TDC ADME](https://tdcommons.ai/single_pred_tasks/adme/) and [MoleculeNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/). No dataset was downloaded or profiled in this session.

| Provisional rank | Dataset / endpoint | Reported size and labels | Why consider it | What blocks commitment |
|---|---|---|---|---|
| 1 | TDC hERG_Karim | 13,445; binary, below 10 micromolar versus at least 10 | Clear screening endpoint; enough data for baseline comparison and scaffold analysis | Audit assay/source aggregation, duplicates, censoring, class balance and release; continuous original values may not survive in prepared labels |
| 2 | TDC AMES | 7,255; mutagenic/nonmutagenic call | Moderate size and operational labels; local motifs make representation comparison plausible | Strain/activation metadata, disagreements, aggregation, analog leakage and prevalence need checking |
| 3 | AqSolDB / TDC Solubility_AqSolDB | 9,982; continuous aqueous solubility, proposed categories not yet defined | Best bridge to regression-versus-classification and feature audit | Mixed conditions, reliability groups and source overlap; a scientific threshold may not be defensible for the full collection |
| 4 | MoleculeNet BACE | Original paper lists 1,522 classification records; commonly processed variants have 1,513 | Small ligand activity task; some versions carry potency and labels | Assay units, class construction, processed counts and test size; ligand-only input omits target interactions |
| 5 | Tox21 | Original MoleculeNet lists 8,014; common prepared version 7,831, 12 assay tasks, missing labels | Later multilabel transfer/robustness study | Release discrepancy, assay-specific imbalance, missingness and documented provenance concerns complicate a first task |
| Supporting only | ESOL / FreeSolv / Lipophilicity | 1,128 / 642 / 4,200 continuous endpoints | Feature-pipeline sanity checks and links to foundation paper | Small samples or no justified categories; FreeSolv is particularly weak as a primary hard-classification application |

All primary candidates provide structural identifiers/SMILES; experimentally matched 3D conformer ensembles are not established as available here. Generated conformers would be an additional modeling assumption. Most first baselines should be feasible on CPU; a standard GNN may benefit from GPU but resource needs require profiling. Do not start by pretraining a foundation model.

**Provisional primary task:** compare representations for identifying hERG assay-defined inhibition liability on unfamiliar chemical series. This is a task recommendation, not a commitment to a GNN or the 35-feature representation. If comparing direct classification with regression is the central scientific question, prefer a continuous assay dataset with reliable original values; hERG's prepared binary labels alone cannot support that comparison. AqSolDB is conditional on adequate experimental metadata and a justified decision threshold.

Stage 5 must still audit actual class frequencies, invalid structures, conflicting duplicates, missing labels/conditions, existing comparable results, licenses and feasible holdouts. Train/test contamination can occur between datasets too; ESOL and larger solubility collections are not automatically independent external validations.

## 6. Progressive learning and experimental roadmap

This is proposed work. Dates and durations are deliberately not assumed without knowing your available time and hardware.

| Step | Learn / do | Deliverable and criterion to continue |
|---|---|---|
| A. Scientific endpoint | ADMET vocabulary, assay interpretation, units, label provenance | One-page task card; another reader can reconstruct the category and explain the scientist's action |
| B. Chemical representation | SMILES, graph connectivity, charge, H, stereochemistry, donors/acceptors | Annotate three molecules by hand; reconcile all 35 feature definitions with code and identify ambiguities |
| C. Data audit | Standardization, duplicates, censoring, label conflicts, class counts | Dataset card, exclusion log and fixed molecule-group identifiers; do not merge incompatible measurements |
| D. Baselines | Logistic regression, tree classifiers, fingerprints/descriptors, cross-validation | Prevalence-only baseline; Morgan radius-2 bit fingerprint (e.g. 2048 bits) + logistic regression and random forest; small 2D descriptors + same models. Radius/length are starting specifications, not optimized facts |
| E. Prediction formulation | BCE, squared/robust regression, cumulative ordinal models | Fixed representation; classifier vs thresholded regression only if continuous labels exist; ordinal comparison only if ordered categories are justified |
| F. Graph comparison | Message passing, bond inputs, permutation-invariant readout | Standard D-MPNN or comparable bond-aware graph baseline; +2D globals with fixed backbone/readout; then compare fusion placement |
| G. Geometry | Conformers, force fields, invariance and partial charges | Rotation/conformer-seed audit; invariant-summary baseline; track failed inputs; proceed only if 3D adds reproducible utility |
| H. Focused contribution | Hypothesis testing, ablations, error analysis | One hypothesis with narrow novelty claim, nearest-work comparison, paired results and explicit trade-offs |

Later evaluation contract:

1. Freeze molecular identity and grouping policy before splitting; deduplicate and keep relevant replicates/analogs from leaking across partitions. Record parent/salt/stereo policies; do not erase chemically relevant differences.
2. Use a locked test set or nested/grouped evaluation. Fit imputation, scaling, feature selection, resampling, probability calibration and prediction thresholds using training/validation only. If calibration needs its own held-out data, budget it explicitly; do not repeatedly reuse the test set.
3. Use scaffold holdout as one stress test, plus time/source or similarity-cluster holdout where metadata allows. Document treatment of acyclic molecules rather than silently placing all into one empty-scaffold bucket. Record nearest-training-compound similarity.
4. Compare methods on identical retained compounds and splits with comparable tuning resources. Use multiple initialization seeds separately from multiple split seeds; changing only network initialization does not measure partition uncertainty.
5. Report paired differences and intervals. Bootstrap molecular/scaffold groups as appropriate; repeated folds are correlated, and 25 runs are not 25 independent datasets. Small tests may be too imprecise to support architecture claims.
6. Predeclare a practically meaningful improvement and primary decision metric with the scientist. Treat secondary metrics and many ablations as exploratory; avoid selecting the best seed or threshold after viewing test results.
7. Preserve quantitative measurements, assay metadata, hard labels, probabilities and review flags. Examine boundary cases, rare classes, activity cliffs, distant scaffolds and preprocessing failures separately.
8. Report inference/preprocessing time, model size, conformer failure rate and review burden alongside predictive metrics. A larger score does not justify a costly representation automatically.

Do not prolong conceptual debate when the next question is empirical. After the task/label audit and baseline protocol are agreed, a small fingerprint-versus-descriptor comparison is more informative than choosing among elaborate architectures on intuition.

## 7. Your first concrete task

**Write a one-page endpoint-and-decision card for hERG_Karim before implementing any model.** Read the [official task definition](https://tdcommons.ai/single_pred_tasks/tox/) and the [CardioTox net source paper](https://link.springer.com/article/10.1186/s13321-021-00541-z), especially dataset construction.

Include:

1. The precise biological endpoint and what a positive label means, including units and equality at the boundary.
2. Where measurements came from and how duplicates, censoring or conflicting sources were combined. Mark “not yet verified” where necessary.
3. One concrete researcher decision supported by the class, and one conclusion the class cannot support.
4. Which mistake is more costly for that decision, why, and how a limited testing budget changes the comparison.
5. Whether original continuous potency values are available. Explain why binary labels alone cannot train a meaningful potency regressor.
6. Your argument for whether 2D structure should be an adequate first representation, and what missing information could defeat it.

Bring back your reasoning rather than a selected architecture. Our next step is to critique the card, audit actual data and settle a falsifiable baseline comparison. The alternate solubility card is worth pursuing if you are most interested in the consequences of discretization.

## 8. Session outcome and limits

Completed: an initial investigation of Stages 1-4, detailed feature inventory and critique, representation/classification comparisons, a structured initial literature matrix, provisional directions, learning roadmap and preserved decision log.

Proposed: task and dataset audit, focused novelty review, baseline experiments and controlled feature/formulation studies.

Not completed or claimed: exhaustive systematic review, full code audit, full-text audit of every recent lead, dataset profiling, model reproduction, validated classification advantage, final category boundaries or a novel research contribution. No performance result in this notebook was generated by our project.
