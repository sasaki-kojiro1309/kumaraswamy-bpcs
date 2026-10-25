# Initial literature matrix

Search cutoff: 24 October 2026. Companion to SESSION_01.md. This is an initial evidence map, not an exhaustive systematic review. Author lists use first author et al. where lengthy; linked primary records contain full lists. “Available” means located/reported, not successfully reproduced. NR means not verified in this session, rather than absent from the paper. This distinction matters especially for abstract-level leads.

## A. Foundation, representations and datasets

### P1. Foundation paper - detailed read

**Citation:** Tetiana Lutchyn, Marie Mardal, Benjamin Ricaud (2025). *Efficient Learning of Molecular Properties Using Graph Neural Networks Enhanced with Chemistry Knowledge*. ACS Omega 10, 54421-54429. [Published paper](https://doi.org/10.1021/acsomega.5c07178); [repository](https://github.com/uitml/TChemGNN).

- **Objective / data:** make a compact graph model effective using supplied chemical information; ESOL, FreeSolv, lipophilicity and quantitative BACE.
- **Input / architecture / formulation:** 14 local +15 molecular +6 geometry-related entries per atom; broadcast global values, GAT layers, pooled or selected-node output; regression, not classification.
- **Methodology / baselines / metrics:** default random 80/10/10, additional cross-validation, feature/layer/readout ablations; GCN variants, descriptor random forest and literature model scores; RMSE.
- **Reported results:** ESOL test 0.7844 and cross-validation 0.7310 +/-0.0835; FreeSolv cross-validation 1.1159 +/-0.1487. These are paper claims, not our reproductions.
- **Limitations:** comparisons need matched splits/cohorts; ambiguous geometry conventions; no direct classification evidence; selected-node interpretation is not causal explanation. Global-feature ablations do not uniquely attribute gains to geometry.
- **Reproducibility / relevance:** public repository located; source preprocessing fetch failed, so no code audit completed. Primary candidate feature inventory and scientific audit target.

### P2. ECFP - foundational method

**Citation:** David Rogers, Mathew Hahn (2010). *Extended-Connectivity Fingerprints*. Journal of Chemical Information and Modeling 50, 742-754. [DOI](https://doi.org/10.1021/ci100050t).

- **Objective / data:** encode chemically useful local environments; not a dataset proposal for our endpoint.
- **Input / architecture / formulation:** atom invariants iteratively expanded into circular neighborhoods; fixed representation for downstream models, not a specific classifier.
- **Methodology / baselines / metrics:** algorithmic and application comparison; exact experimental settings/metrics NR here.
- **Result / limitation:** established representation family; radius, hashing/count choices and chirality settings limit what is retained.
- **Reproducibility / relevance:** contemporary toolkit implementations exist, but identical implementation to the original method must not be presumed. Essential inexpensive structural baseline. Reading depth: metadata and method overview verified, not full experimental audit.

### P3. Message passing - foundational framework

**Citation:** Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl (2017). *Neural Message Passing for Quantum Chemistry*. ICML, PMLR 70, 1263-1272. [Proceedings](https://proceedings.mlr.press/v70/gilmer17a).

- **Objective / data:** unify graph models and predict quantum molecular properties; QM9 experiments.
- **Input / architecture / formulation:** atom, bond and spatial information; learned message functions, node updates and readout; continuous-property prediction.
- **Methodology / baselines / metrics:** variants of message passing and earlier quantum-property predictors; property-specific prediction errors. Exact split/code audit NR.
- **Result / limitation:** provides a framework for symmetry-respecting molecular learning, not proof that any GNN beats fingerprints on hard labels.
- **Reproducibility / relevance:** public proceedings verified; code availability not independently audited. Defines a graph baseline's conceptual components. Reading depth: primary framework description.

### P4. Learned representations and descriptor integration - central prior work

**Citation:** Kevin Yang et al. (2019). *Analyzing Learned Molecular Representations for Property Prediction*. JCIM 59, 3370-3388. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6727618/).

- **Objective / data:** compare representations across 19 public and 16 industrial datasets.
- **Input / architecture / formulation:** directed bond message passing, optionally RDKit global descriptors; classification and regression.
- **Methodology / baselines / metrics:** split/tuning and representation comparisons against fixed descriptors/fingerprints and prior graph methods; task-specific AUC and regression errors.
- **Result / limitation:** added global features help some datasets and harm others; industrial data access restricts full replication. Results are conditional, not universal.
- **Reproducibility / relevance:** Chemprop code linked by authors; not executed here. Direct precedent against claiming graph/descriptor fusion as new. Reading depth: methods and descriptor-effect results inspected.

### P5. MoleculeNet - benchmark provenance

**Citation:** Zhenqin Wu et al. (2018). *MoleculeNet: a benchmark for molecular machine learning*. Chemical Science 9, 513-530. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/).

- **Objective / data:** benchmark diverse quantum, physicochemical, biological and physiological tasks.
- **Input / architecture / formulation:** structures and sometimes coordinates; classical and learned representations; regression, binary and multitask classification.
- **Methodology / baselines / metrics:** dataset-specific random/scaffold/task splits and diverse classical/graph methods; RMSE/MAE, AUROC or PR metrics as specified.
- **Result / limitation:** supplies common tasks, not a universal best model or universally realistic split; downstream processed releases differ.
- **Reproducibility / relevance:** datasets/loaders available through benchmark ecosystem, not profiled here. Source of candidate endpoints and original counts. Reading depth: dataset and protocol sections.

### P6. Therapeutics Data Commons - task formulation resource

**Citation:** Kexin Huang et al. (2021). *Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development*. NeurIPS Datasets and Benchmarks. [Proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/4c56ff4ce4aaf9573aa5dff913df997a-Abstract-round1.html).

- **Objective / data:** standardize access/evaluation across therapeutics tasks, including ADMET.
- **Input / architecture / formulation:** task-dependent molecules and other biomedical entities; no single architecture or prediction type.
- **Methodology / baselines / metrics:** task-specific benchmarks, split utilities and evaluation; exact endpoint protocols require current documentation.
- **Result / limitation:** enables access and comparison while highlighting unsolved generalization; API-ready does not imply chemically homogeneous labels.
- **Reproducibility / relevance:** open library and task documentation; no loader run here. Candidate dataset registry. Reading depth: original record and current ADME/toxicity task pages.

### P7. AqSolDB - continuous-label candidate

**Citation:** M. C. Sorkun, A. Khetan, S. Er (2019). *AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds*. Scientific Data 6, 143. [DOI](https://doi.org/10.1038/s41597-019-0151-1); [authors' institutional record](https://research.tue.nl/en/publications/aqsoldb-a-curated-reference-set-of-aqueous-solubility-and-2d-desc/).

- **Objective / data:** curate 9,982 compounds from nine solubility collections.
- **Input / architecture / formulation:** structures, continuous solubility, 2D descriptors and reliability grouping; dataset construction rather than classifier proposal.
- **Methodology / baselines / metrics:** identifier/unit curation and reliability annotation; predictive baseline results NR here.
- **Result / limitation:** larger curated resource, but heterogeneous experimental sources/conditions remain relevant to category validity.
- **Reproducibility / relevance:** public data and TDC access documented; no row-level audit. Strong potential for direct-versus-regression comparison conditional on category justification. Reading depth: official/institutional description; publisher full-text access failed.

### P8. CardioTox net - hERG source model and data

**Citation:** Abdul Karim et al. (2021). *CardioTox net: a robust predictor for hERG channel blockade based on deep learning meta-feature ensembles*. Journal of Cheminformatics 13, 60. [Primary paper](https://link.springer.com/article/10.1186/s13321-021-00541-z).

- **Objective / data:** hERG inhibitor screening using curated bioactivity collections and several test sets.
- **Input / architecture / formulation:** molecular graphs, fingerprints, descriptors and learned features; ensemble binary classifier.
- **Methodology / baselines / metrics:** comparisons with hERG predictors and multiple test cohorts; accuracy, balanced accuracy, sensitivity and related classification statistics.
- **Result / limitation:** reports competitive screening prediction; assay heterogeneity and applicability limit interpretation as a clinical safety assessment. No claim of superior utility for our project is accepted.
- **Reproducibility / relevance:** source repository reported as [CardioTox](https://github.com/Abduk084/CardioTox); not run/audited. Nearest endpoint-specific source for the first student task. Reading depth: primary article and task provenance overview; detailed curation audit remains.

### P9. Uni-Mol - geometry-aware pretrained comparator

**Citation:** Gengmo Zhou et al. (2023). *Uni-Mol: A Universal 3D Molecular Representation Learning Framework*. ICLR 2023. [Paper](https://openreview.net/pdf?id=6K2RM6wVqKu); [official repository](https://github.com/deepmodeling/Uni-Mol).

- **Objective / data:** transferable 3D molecular/pocket representations; large structural pretraining and downstream benchmarks.
- **Input / architecture / formulation:** atom and pairwise spatial information in a transformer; pretraining followed by task heads, including classification/regression.
- **Methodology / baselines / metrics:** property and 3D tasks compared with prior representation methods; dataset-specific AUC/error measures.
- **Result / limitation:** demonstrates geometry-aware transfer in its settings; greater resource needs and input/pretraining comparability complicate small-data comparisons.
- **Reproducibility / relevance:** official code located, not executed. A later comparator, not required for first experiments. Reading depth: primary overview and official repository; downstream protocol not fully audited.

### P10. MARCEL - conformer ensembles

**Citation:** Yanqiao Zhu et al. (2024). *Learning Over Molecular Conformer Ensembles: Datasets and Benchmarks*. ICLR 2024. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7eb6233e02f7d9efbb84acd839a996fb-Abstract-Conference.html).

- **Objective / data:** four datasets of ensemble-dependent molecular/reaction properties, including catalysts.
- **Input / architecture / formulation:** strings, 2D graphs, single 3D structures and conformer ensembles; primarily quantitative targets.
- **Methodology / baselines / metrics:** representative 1D/2D/3D models and two ensemble integration approaches; task-specific property errors.
- **Result / limitation:** ensemble learning can improve examined tasks; does not prove gains for arbitrary ADMET hard categories or inexpensive generated conformers.
- **Reproducibility / relevance:** [project/code](https://github.com/SXKDZ/MARCEL) listed by paper; not run. Prior art and warning against single-conformer assumptions. Reading depth: abstract plus primary paper overview; tables not exhaustively audited.

## B. Classification, calibration and failure modes

### P11. CORAL - ordered categorical objectives

**Citation:** Wenzhi Cao, Vahid Mirjalili, Sebastian Raschka (2020). *Rank consistent ordinal regression for neural networks with application to age estimation*. Pattern Recognition Letters 140, 325-331. [Authors' paper](https://sebastianraschka.com/pdf/papers/2020-coral.pdf).

- **Objective / data:** consistent ordinal probabilities on facial age datasets.
- **Input / architecture / formulation:** image encoder with cumulative binary ordinal tasks and constrained shared structure.
- **Methodology / baselines / metrics:** comparisons with reference ordinal networks; age error measures.
- **Result / limitation:** theoretical rank consistency and reported age-estimation gains; no molecular evidence.
- **Reproducibility / relevance:** paper available; implementation not audited. Candidate objective only for defensibly ordered classes. Reading depth: method/abstract and publication metadata.

### P12. Calibration of neural classifiers

**Citation:** Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger (2017). *On Calibration of Modern Neural Networks*. ICML, PMLR 70, 1321-1330. [Proceedings](https://proceedings.mlr.press/v70/guo17a.html).

- **Objective / data:** characterize probability miscalibration in image/document classification.
- **Input / architecture / formulation:** neural classifiers and posthoc calibration, including temperature scaling.
- **Methodology / baselines / metrics:** compare calibration methods and network settings; calibration and predictive scores.
- **Result / limitation:** temperature scaling often effective in examined settings; chemical/scaffold-shift validity not established.
- **Reproducibility / relevance:** method is simple to implement later; source not run. Motivates calibrated probabilities alongside hard labels. Reading depth: abstract and primary record.

### P13. Conformal prediction foundations

**Citation:** Anastasios N. Angelopoulos, Stephen Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv 2107.07511, first 2021, accessed v6 (2022). [Author preprint](https://arxiv.org/abs/2107.07511).

- **Objective / data:** explain prediction sets/intervals and assumptions; methodological tutorial, not one molecular benchmark.
- **Input / architecture / formulation:** model-agnostic conformity scores plus calibration observations.
- **Methodology / baselines / metrics:** coverage and set/interval size; theoretical guarantees under stated assumptions.
- **Result / limitation:** marginal guarantees under exchangeability do not imply per-molecule or arbitrary out-of-distribution validity.
- **Reproducibility / relevance:** examples described in paper; code not audited. Supports a review/uncertainty layer without redefining chemical classes. Reading depth: framework overview.

### P14. MoleculeACE - difficult local structure-property behavior

**Citation:** Derek van Tilborg, Alisa Alenicheva, Francesca Grisoni (2022). *Exposing the Limitations of Molecular Machine Learning with Activity Cliffs*. JCIM 62, 5938-5951. [Primary paper](https://doi.org/10.1021/acs.jcim.2c01073); [authors' record](https://research.tue.nl/en/publications/exposing-the-limitations-of-molecular-machine-learning-with-activ/).

- **Objective / data:** compare prediction on activity-cliff compounds across 30 target-specific potency datasets.
- **Input / architecture / formulation:** descriptor/fingerprint and deep representations; potency regression, with cliff-centered evaluation.
- **Methodology / baselines / metrics:** multiple classical/deep models, overall and cliff-specific RMSE.
- **Result / limitation:** difficult cliff behavior and strong classical baselines; not a direct test of our category boundaries.
- **Reproducibility / relevance:** [MoleculeACE code/data](https://github.com/molML/MoleculeACE) available; [published correction](https://pubmed.ncbi.nlm.nih.gov/36995229/) located but correction content not audited. Use corrected assets before reproduction. Reading depth: primary abstract/results overview and official repository.

### P15. Tox21 provenance and reproducibility

**Citation:** Antonia Ebner, Christoph Bartmann, Sonja Topf, Sohvi Luukkonen, Johannes Schimunek, Günter Klambauer (2025). *Measuring AI Progress in Drug Discovery: A Reproducible Leaderboard for the Tox21 Challenge*. arXiv 2511.14744v1, 18 November 2025. [Primary preprint](https://arxiv.org/html/2511.14744v1).

- **Objective / data:** restore comparable evaluation on original Tox21 Challenge data.
- **Input / architecture / formulation:** descriptors/fingerprints and neural/graph comparators; assay classification.
- **Methodology / baselines / metrics:** original challenge setting and reproducible leaderboard; includes classical and learned baselines. Exact metric aggregation/splits not audited here.
- **Result / limitation:** authors report altered labels across dataset integrations and continued competitiveness of earlier methods. These claims need release-specific inspection before applying them to our download.
- **Reproducibility / relevance:** public [leaderboard](https://huggingface.co/spaces/ml-jku/tox21_leaderboard) and inference resources reported, not tested. Strong reason to audit dataset identity. Reading depth: abstract and article structure; not full replication audit. Peer-reviewed venue not verified.

## C. Recent leads and novelty checks - do not treat as reproduced evidence

| Record | Objective, inputs and formulation | Experiments, baselines, metrics and reported conclusion | Limitations, code status and relevance |
|---|---|---|---|
| Jiacheng Zheng, Chang Guo, Zixuan Wang, Xinyu Liu, Hao Chen. *Beyond Scaffold Splits: Structural-Frontier Evaluation Reveals Hidden Failures in ADMET Models*. [arXiv 2607.10729v4](https://arxiv.org/abs/2607.10729v4), revised 6 October 2026; peer-reviewed venue not verified | Structural-distance evaluation on six ADMET tasks; molecular views and graph/perceptron controls; mixed endpoint tasks | Scaffold versus frontier holds; Lo-Hi/DataSAIL comparators; normalized task errors. Authors report substantial degradation and inconclusive robust-training gains | Abstract/version history checked. Exact code, protocol and numerical results not independently audited. Counters any claim that scaffold holdout alone resolves novelty/generalization |
| Hyeonsu Lee, Juyeon Kim, Erkhembayar Jadamba, Seungjin Choi, Hyunjin Shin. *Conformal Prediction for Molecular Properties under Label Shift*. [arXiv 2608.17678v1](https://arxiv.org/html/2608.17678v1), 18 August 2026; venue not independently verified | AqSolDB regression with chemical-language model and weighted conformal intervals | Label-shift scenarios; BBSE/RLLS/MLE weight estimates; interval coverage/width. Proposed benefit is shift-adapted uncertainty, not hard category accuracy | Methods inspected. Assumes stable P(x|y); estimated weighting and binned continuous targets require scrutiny; code NR. Nearest uncertainty prior work for solubility, not a license to claim arbitrary-shift guarantees |
| Bryan Cheng, Austin Jin, Jasper Zhang (2026). arXiv record title: *When Three-Dimensional Conformer Ensembles Improve Molecular Property Prediction Beyond Two-Dimensional Fingerprints: A Systematic Study*. [Record](https://arxiv.org/abs/2606.08825); [HTML](https://arxiv.org/html/2606.08825v1) uses title *When Does Conformer Geometry Help? Complementarity of 3D Ensemble Statistics and 2D Fingerprints for Molecular Property Prediction* | Fingerprints plus generated ensemble summaries; MoleculeNet/QM9/MARCEL; 14 regression and two classification targets according to authors | Compares Morgan/XGBoost, ensemble feature methods and 3D networks; RMSE and task-specific classification measures. Reports property-dependent complementarity | Record/HTML metadata discrepancy explicitly unresolved. Record claims ACM-BCB 2026; proceedings not independently verified. Numerical claims and code unaudited. Prior-work lead for geometry worth checking before novelty claims |

Additional verified-title leads requiring full matrix entries in a later focused review:

- [A systematic study of key elements underlying molecular property prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC10575948/) (Nature Communications, 2023): representation/split/descriptor controls; full bibliographic and protocol extraction pending.
- [UMAP-based clustering split for rigorous evaluation of AI models for virtual screening on cancer cell lines](https://link.springer.com/article/10.1186/s13321-025-01039-8) (Journal of Cheminformatics, 2025): NCI-60 and screening-oriented split evaluation; source-specific results cannot be generalized to all endpoints without testing.
- [Symmetry-Preserving Conformer Ensemble Networks for Molecular Representation Learning](https://papers.neurips.cc/paper_files/paper/2025/hash/62891522c00cf7323cbacb500e6cfc8d-Abstract-Conference.html) (NeurIPS 2025): confirms ensemble symmetry is active prior art; full methods/code audit pending.
- [Activity-cliff awareness enables robust graph learning for molecular property prediction](https://www.nature.com/articles/s41467-026-75713-2) (Nature Communications 17, 9376, 2026): primary indexed record found; full text access failed, so model/results not used to establish conclusions here.

## Search continuation protocol

After selecting the endpoint/hypothesis: inspect supplements, corrections and code of the nearest studies; record dataset versions, split identities, retained cohorts, tuning budget, label definition and uncertainty intervals. Search forward citations from P1 and P4 and endpoint-specific papers using a citation index, then log inclusion/exclusion reasons. Verify recent venue claims against proceedings, not only author comments. Record counterevidence and null results with the same care as positive results. A missing item in this initial search is not evidence that the corresponding research gap is novel.
