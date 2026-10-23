# Stage 2: block progressive censoring

This milestone implements the experimental design and validates completed data.
It does not generate lifetimes, select withdrawals, construct a likelihood, or
estimate parameters.

## Source and Kumaraswamy adaptation

**Reference 1 — Wang et al. (2023)**, Section 2.1, Eq. (4), printed p.2741 /
PDF p.6, organizes k facilities. Facility i begins with n_i units, a prefixed
failure target m_i, and a withdrawal schedule R_i=(r_i1,...,r_i,m_i).
The observed row is (x_ij,delta_ij,r_ij). Each facility conducts its own progressive
Type-II experiment. Their failure sequences are not merged into one stopping rule.

This organization is independent of whether latent lifetimes follow IEED or
Kumaraswamy distributions. Our adaptation retains the agreed independent
Kw(a,b_i1), Kw(a,b_i2) lifetimes and requires 0<x_ij<1. We use no IEED formula.
Our fixed shared a and facility-specific b values are the user's provisional
parameter choices, as documented in MODEL_SPEC.md.

## What happens at each failure?

1. Fix n_i, m_i, and all withdrawals before testing. Counts are nonnegative
   integers; 1<=m_i<=n_i. A completed block satisfies n_i=m_i+sum_j r_ij.
2. Immediately before failure j, count units still alive and under test:
   \[
   Y_{ij}=n_i-(j-1)-\sum_{\ell<j}r_{i\ell}. \tag{C1}
   \]
   The subtraction removes the preceding j-1 failures and all previous withdrawals.
3. Record one failure time x_ij and cause delta_ij in {1,2}. This leaves Y_ij-1
   survivors, so the planned r_ij cannot exceed Y_ij-1.
4. Randomly remove r_ij survivors, independently of their unobserved failure
   times and latent causes. The number left after this row is
   \[
   L_{ij}=Y_{ij}-1-r_{ij}. \tag{C2}
   \]
   For a nonfinal row, this is Y_i,j+1. It must leave at least one unit for the
   next planned failure. The final row leaves zero survivors under test.
5. Stop that block at its m_i-th failure. Blocks can end at different times.

The risk set counts units still under test, not all units still alive anywhere.
Withdrawn units may remain alive outside the experiment but leave its risk set.
Counting all units gives n_i=m_i+sum_j r_ij. Conversely, with nonnegative counts
and that total, (C1) equals sum_{ell=j}^{m_i}(1+r_i,ell), so every required failure
has a unit available. The implementation checks both totals and row transitions
to provide useful error messages.

## Why a withdrawal contains statistical information

A survivor removed at time t is right censored: its unobserved lifetime exceeds
t. In our model this event has probability S_i(t)=(1-t^a)^(b_i1+b_i2).
For r withdrawn units, independence gives the survival event probability S_i(t)^r.
Dropping the count would discard information about units known to have survived.
They are neither failures at t nor observations of their eventual failure causes.
This explains the role of withdrawals; no full likelihood is constructed here.

The table can check counts, not whether the actual removals were random.
Noninformative survivor selection remains an experimental assumption. Selecting
the apparently weakest units, for example, would need a different observation
model. No withdrawal-selection algorithm has been implemented in this milestone.

## Data interfaces

Implementation: `../src/censoring.py`.

| Interface | Meaning and validation |
| --- | --- |
| `BlockScheme(block,n,m,withdrawals)` | Immutable planned scheme; validates integer counts, schedule length, all risk transitions, and completed-block total |
| `.risk_sets` | Tuple of Y_ij values immediately before failures |
| `Observation(block,time,cause,withdrawals)` | Immutable row with finite time in (0,1), cause 1 or 2, and integer withdrawal count |
| `BlockData(scheme,observations)` | Exactly m rows; validates block identity, strictly increasing times, known causes, and each withdrawal against the planned schedule |
| `.cause_counts` | Counts of observed cause-1 and cause-2 failures; zeros are permitted |
| `BPCSExperiment(blocks)` | Nonempty collection with unique block IDs; supports any positive block count |
| `.to_frame()` | Fresh pandas table with observation fields, failure index, risk set, survivors after failure, and remaining after withdrawals |
| `.summary_frame()` | Fresh per-block sizes, total withdrawals, and observed cause counts |

Block IDs are nonempty strings. Integer counts must be actual integer values;
booleans and floating-point counts are rejected rather than silently coerced.
NumPy integer scalars are accepted. Inputs supplied as lists become tuples so
later edits cannot invalidate the validated objects. Editing exported tables
does not mutate the source objects.

Within-block times must already be strictly increasing. We never silently sort:
sorting can hide mismatches between withdrawals and their planned failure index.
Ties are rejected within blocks under the current continuous exact-time model.
Equal or overlapping times across different facilities are allowed. Rounded
within-block ties require a later justified measurement model.

`Observation.cause=None` reserves a missing-cause representation, but `BlockData`
rejects it in the known-cause baseline. Code 0 is not introduced as a third cause.
Allowing missing causes in analysis requires a future model decision.
The data layer deliberately accepts blocks with no failures from one cause:
data validity and existence of an interior MLE are different questions.

Parameters are not stored in observed data. They belong to `CompetingRisks` and
later estimation interfaces, allowing many candidate parameter values to be
evaluated on the same validated experiment.

## Two-block teaching example

The hand-constructed rows in `../data/examples/two_blocks.json` are neither
simulated nor real observations. Run from the project root:

```sh
.venv/bin/python -m experiments.stage2_example
```

| Block | j | Failure time | Cause | Y: before failure | Survivors after failure | Withdrawals r | L: remaining |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 1 | .15 | 1 | 6 | 5 | 1 | 4 |
| A | 2 | .40 | 2 | 4 | 3 | 0 | 3 |
| A | 3 | .70 | 1 | 3 | 2 | 2 | 0 |
| B | 1 | .10 | 2 | 7 | 6 | 0 | 6 |
| B | 2 | .25 | 1 | 6 | 5 | 1 | 4 |
| B | 3 | .55 | 2 | 4 | 3 | 0 | 3 |
| B | 4 | .80 | 2 | 3 | 2 | 2 | 0 |

Block A: n=6, m=3, R=(1,0,2). Start with 6; the first failure and withdrawal
leave 4. The second failure alone leaves 3. The third failure leaves 2, both
withdrawn, ending the block. Thus 6=3+1+0+2.

Block B: n=7, m=4, R=(0,1,0,2). The first failure leaves 6; the second failure
and one withdrawal leave 4; the third failure leaves 3; the fourth failure and
two withdrawals empty the block. Thus 7=4+0+1+0+2.

The combined experiment begins with 13 units, observes 7 failures, and withdraws
6 survivors. Cause counts are (2,1) in A and (1,3) in B. Different times and
different schedules are valid; time .10 in B need not follow the last time in A.

The example script also calls the existing Stage 1 functions with chosen values
a=2, (b_A1,b_A2)=(1,2), (b_B1,b_B2)=(.8,1.2). It gives S_A(.5)=.421875,
S_B(.5)=.5625, and eventual cause-1 probabilities 1/3 and .4. These are
illustrations of model probabilities, not estimates from these seven failures.

## Verification scope

`../tests/test_censoring.py` checks the complete example against the table above,
ordinary Type-II and complete-data special cases, one-failure designs, every
nonnegative completed schedule for n<=6, incorrect counts/totals, excessive or
premature withdrawals, unsupported causes, invalid times, order/tie errors,
row/scheme mismatches, cross-block overlaps, arbitrary block counts, single-cause
blocks, and immutability. The Stage 1 tests remain unchanged.

Validation confirms structural consistency. It cannot establish that hand-chosen
times were generated from a particular distribution or that real withdrawals
were selected noninformatively. No likelihood, optimizer, simulation generator,
Bayesian method, or goodness-of-fit procedure has been added.
