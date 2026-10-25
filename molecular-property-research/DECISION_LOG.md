# Molecular property classification: running decision log

## 24 October 2026 - Session 1

### Authorized scope

Initial literature-based Stages 1-4; particular depth in scientific use and representations. No model implementation. User instructions originate from the pasted research brief; the supplied PDF is a reference.

### Decisions made

- Keep representation and classification formulation as separate factors before testing interactions.
- Treat TChemGNN's 35 values per atom as a candidate, not a complete or optimal feature set.
- Preserve original continuous measurements and metadata when available, even though hard categories are the primary output.
- Distinguish endpoint thresholds from probability decision thresholds.
- Keep direct classification and regression-derived decisions as competing approaches where data support both.
- Provisional primary task for further audit: hERG assay-defined inhibition liability. This is not a final dataset/architecture choice.
- Use an isolated molecular-property-research folder because the existing workspace contains a different statistics project. No existing project files changed.

### Rejected assumptions and rationale

| Assumption | Why rejected |
|---|---|
| The foundation paper proves hard-classification gains | Its experiments are regression |
| All 35 entries are independent and useful | Suspected donor duplication, deterministic element proxies and aggregate redundancy |
| Generated geometry is measured geometry | Embedding/force-field results depend on conformer and settings |
| Absolute molecular orientation should predict an isolated-molecule scalar | Rigid rotation should not change such a property |
| First/last SMILES position proves chemical causality | Serialization varies, cycles have no terminal atom, and global inputs are broadcast to each node |
| A low/medium/high split is inherently scientific | A task and boundary justification are required |
| Scaffold split establishes prospective reliability | It measures one notion of chemical separation; source/time and similarity shifts differ |
| Descriptor concatenation, conformer ensembles or calibration alone are new | All have substantial prior work |

### Outstanding questions

1. Which researcher action and error costs should define the primary endpoint?
2. Does the chosen release retain continuous potency, censoring and assay/source metadata?
3. What are the class counts, duplicate/conflict rates and useful scaffold/source holdouts after parsing?
4. Are donor features 19 and 26 identical in code? What do valence, aromaticity, sp2 fraction and hydroxyl flags mean exactly?
5. How are conformers, partial charges, dipole units/origin and orientation computed?
6. Are paper comparisons reproducible on shared split IDs and matched retained molecules?
7. Which narrow hypothesis remains distinct after checking the nearest papers, corrections and supplements?
8. What time, compute and chemical expertise can the student/professor supply?

### Next personally owned task

Student: write the one-page hERG endpoint-and-decision card in SESSION_01.md Section 7. No training code yet.

### Evidence required before experiments

An endpoint card, checked category rule, dataset provenance/curation plan, baseline specifications, fixed evaluation protocol and one falsifiable question. No novelty or performance claim is currently accepted.

### Research continuity

Append future dated entries; preserve changed decisions, negative results and source revisions. SESSION_01.md contains the scientific investigation; LITERATURE_MATRIX.md records evidence and reading depth. The attached brief is preserved as RESEARCH_BRIEF.txt.
