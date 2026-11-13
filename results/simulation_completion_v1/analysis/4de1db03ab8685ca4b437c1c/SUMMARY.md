# Verified local simulation completion

Selected valid fits: 360/360; missing: 0; invalid: 0.

Original 60 datasets and all historical records are preserved. Selection uses the historical valid result first, then the first passing effort level. The completion registry explicitly combines provenance across campaigns with identical priors/model/data but different implementation and sampling effort. No seed selection or threshold relaxation is used.

| Method | Strict | Equality | Violation | Total |
|---|---:|---:|---:|---:|
| MLE U | 20 | 20 | 20 | 60 |
| MLE R | 20 | 20 | 20 | 60 |
| Independent U | 20 | 20 | 20 | 60 |
| Independent R | 20 | 20 | 20 | 60 |
| Hierarchical U | 20 | 20 | 20 | 60 |
| Hierarchical R | 20 | 20 | 20 | 60 |

## strict_order_late

| Method | Parameter | n paired | AB | MSE |
|---|---|---:|---:|---:|
| MLE U | a | 20 | 0.099645254 | 0.051451993 |
| MLE U | b1[A] | 20 | 0.067722395 | 0.045748547 |
| MLE U | b2[A] | 20 | 0.21204154 | 0.36995385 |
| MLE U | b1[B] | 20 | 0.15094779 | 0.12894833 |
| MLE U | b2[B] | 20 | 0.55555807 | 0.94967221 |
| MLE R | a | 20 | 0.099645254 | 0.051451993 |
| MLE R | b1[A] | 20 | 0.067722395 | 0.045748547 |
| MLE R | b2[A] | 20 | 0.21204154 | 0.36995385 |
| MLE R | b1[B] | 20 | 0.15094779 | 0.12894833 |
| MLE R | b2[B] | 20 | 0.55555807 | 0.94967221 |
| Independent U | a | 20 | 0.090574302 | 0.049317222 |
| Independent U | b1[A] | 20 | 0.072589143 | 0.047925251 |
| Independent U | b2[A] | 20 | 0.22939232 | 0.39323939 |
| Independent U | b1[B] | 20 | 0.16939545 | 0.14090649 |
| Independent U | b2[B] | 20 | 0.61507229 | 1.0716433 |
| Independent R | a | 20 | 0.090213183 | 0.049214606 |
| Independent R | b1[A] | 20 | 0.06575768 | 0.04317277 |
| Independent R | b2[A] | 20 | 0.23651122 | 0.38950313 |
| Independent R | b1[B] | 20 | 0.15893861 | 0.13502051 |
| Independent R | b2[B] | 20 | 0.61923594 | 1.0624741 |
| Hierarchical U | a | 20 | 0.071780438 | 0.036385572 |
| Hierarchical U | b1[A] | 20 | 0.1008856 | 0.045906053 |
| Hierarchical U | b2[A] | 20 | 0.20379883 | 0.31493793 |
| Hierarchical U | b1[B] | 20 | 0.15266277 | 0.10517382 |
| Hierarchical U | b2[B] | 20 | 0.44086513 | 0.64263261 |
| Hierarchical R | a | 20 | 0.075417449 | 0.03675019 |
| Hierarchical R | b1[A] | 20 | 0.095180159 | 0.043125025 |
| Hierarchical R | b2[A] | 20 | 0.22001258 | 0.31849519 |
| Hierarchical R | b1[B] | 20 | 0.1485319 | 0.10355861 |
| Hierarchical R | b2[B] | 20 | 0.46457759 | 0.65273425 |

| Method | Parameter | Interval | n | CP | AW |
|---|---|---|---:|---:|---:|
| Hierarchical U | a | credible | 20 | 0.95 | 0.79072594 |
| Hierarchical U | b1[A] | credible | 20 | 1 | 0.87433806 |
| Hierarchical U | b1[B] | credible | 20 | 0.95 | 1.3919527 |
| Hierarchical U | b2[A] | credible | 20 | 0.95 | 1.8549398 |
| Hierarchical U | b2[B] | credible | 20 | 0.95 | 3.2139 |
| Hierarchical R | a | credible | 20 | 0.95 | 0.79765984 |
| Hierarchical R | b1[A] | credible | 20 | 1 | 0.86005181 |
| Hierarchical R | b1[B] | credible | 20 | 0.95 | 1.3747382 |
| Hierarchical R | b2[A] | credible | 20 | 0.9 | 1.8691885 |
| Hierarchical R | b2[B] | credible | 20 | 0.95 | 3.2349125 |
| Independent U | a | credible | 20 | 0.95 | 0.85198284 |
| Independent U | b1[A] | credible | 20 | 1 | 0.88678097 |
| Independent U | b1[B] | credible | 20 | 0.95 | 1.5326214 |
| Independent U | b2[A] | credible | 20 | 0.9 | 1.9860772 |
| Independent U | b2[B] | credible | 20 | 0.95 | 3.6848301 |
| Independent R | a | credible | 20 | 0.95 | 0.85104117 |
| Independent R | b1[A] | credible | 20 | 1 | 0.86870633 |
| Independent R | b1[B] | credible | 20 | 0.95 | 1.5080074 |
| Independent R | b2[A] | credible | 20 | 0.9 | 1.9781445 |
| Independent R | b2[B] | credible | 20 | 0.95 | 3.662375 |
| MLE U | a | confidence | 20 | 0.95 | 0.87150601 |
| MLE U | b1[A] | confidence | 20 | 0.95 | 1.0519752 |
| MLE U | b1[B] | confidence | 20 | 0.95 | 1.8345911 |
| MLE U | b2[A] | confidence | 20 | 0.9 | 2.1521804 |
| MLE U | b2[B] | confidence | 20 | 1 | 4.0278047 |
| MLE R | a | confidence | 20 | 0.95 | 0.87150601 |
| MLE R | b1[A] | confidence | 0 | NA | NA |
| MLE R | b1[B] | confidence | 0 | NA | NA |
| MLE R | b2[A] | confidence | 0 | NA | NA |
| MLE R | b2[B] | confidence | 0 | NA | NA |

## equality_late

| Method | Parameter | n paired | AB | MSE |
|---|---|---:|---:|---:|
| MLE U | a | 20 | 0.0040328987 | 0.037729916 |
| MLE U | b1[A] | 20 | 0.014240588 | 0.12649302 |
| MLE U | b2[A] | 20 | 0.056454669 | 0.182972 |
| MLE U | b1[B] | 20 | 0.061061979 | 0.3713545 |
| MLE U | b2[B] | 20 | 0.067459095 | 0.17491818 |
| MLE R | a | 20 | 0.0040328987 | 0.037729916 |
| MLE R | b1[A] | 20 | 0.08165306 | 0.096261201 |
| MLE R | b2[A] | 20 | 0.12386714 | 0.19039407 |
| MLE R | b1[B] | 20 | 0.028251408 | 0.26137549 |
| MLE R | b2[B] | 20 | 0.15677248 | 0.23245345 |
| Independent U | a | 20 | 0.0048221168 | 0.037175663 |
| Independent U | b1[A] | 20 | 0.0056849474 | 0.13125976 |
| Independent U | b2[A] | 20 | 0.066395543 | 0.19359387 |
| Independent U | b1[B] | 20 | 0.088924527 | 0.40199954 |
| Independent U | b2[B] | 20 | 0.093642042 | 0.19154904 |
| Independent R | a | 20 | 0.0051171256 | 0.037293475 |
| Independent R | b1[A] | 20 | 0.16446274 | 0.098249232 |
| Independent R | b2[A] | 20 | 0.22326851 | 0.23134983 |
| Independent R | b1[B] | 20 | 0.16070302 | 0.24192236 |
| Independent R | b2[B] | 20 | 0.34195792 | 0.3911886 |
| Hierarchical U | a | 20 | 0.012988017 | 0.02881166 |
| Hierarchical U | b1[A] | 20 | 0.011301628 | 0.11129156 |
| Hierarchical U | b2[A] | 20 | 0.078076103 | 0.16927035 |
| Hierarchical U | b1[B] | 20 | 0.0281987 | 0.2587993 |
| Hierarchical U | b2[B] | 20 | 0.043036319 | 0.12131292 |
| Hierarchical R | a | 20 | 0.011571029 | 0.028689604 |
| Hierarchical R | b1[A] | 20 | 0.14522628 | 0.076523978 |
| Hierarchical R | b2[A] | 20 | 0.23185807 | 0.20137184 |
| Hierarchical R | b1[B] | 20 | 0.20086987 | 0.17394184 |
| Hierarchical R | b2[B] | 20 | 0.27895866 | 0.24012569 |

| Method | Parameter | Interval | n | CP | AW |
|---|---|---|---:|---:|---:|
| Hierarchical U | a | credible | 20 | 1 | 0.7579343 |
| Hierarchical U | b1[A] | credible | 20 | 0.95 | 1.2069107 |
| Hierarchical U | b1[B] | credible | 20 | 0.9 | 1.9343932 |
| Hierarchical U | b2[A] | credible | 20 | 0.9 | 1.2726149 |
| Hierarchical U | b2[B] | credible | 20 | 1 | 1.9475668 |
| Hierarchical R | a | credible | 20 | 1 | 0.7664352 |
| Hierarchical R | b1[A] | credible | 20 | 0.85 | 0.96524645 |
| Hierarchical R | b1[B] | credible | 20 | 0.85 | 1.5586412 |
| Hierarchical R | b2[A] | credible | 20 | 0.95 | 1.3043853 |
| Hierarchical R | b2[B] | credible | 20 | 1 | 2.034032 |
| Independent U | a | credible | 20 | 0.95 | 0.81627949 |
| Independent U | b1[A] | credible | 20 | 0.95 | 1.2548326 |
| Independent U | b1[B] | credible | 20 | 0.9 | 2.1723755 |
| Independent U | b2[A] | credible | 20 | 0.9 | 1.3327289 |
| Independent U | b2[B] | credible | 20 | 1 | 2.160136 |
| Independent R | a | credible | 20 | 1 | 0.81667088 |
| Independent R | b1[A] | credible | 20 | 0.85 | 0.99427213 |
| Independent R | b1[B] | credible | 20 | 0.9 | 1.7253354 |
| Independent R | b2[A] | credible | 20 | 0.95 | 1.3682118 |
| Independent R | b2[B] | credible | 20 | 1 | 2.2677031 |
| MLE U | a | confidence | 20 | 1 | 0.83354016 |
| MLE U | b1[A] | confidence | 20 | 0.95 | 1.3905817 |
| MLE U | b1[B] | confidence | 20 | 0.95 | 2.4151818 |
| MLE U | b2[A] | confidence | 20 | 0.95 | 1.464062 |
| MLE U | b2[B] | confidence | 20 | 1 | 2.4036188 |
| MLE R | a | confidence | 20 | 1 | 0.83354016 |
| MLE R | b1[A] | confidence | 0 | NA | NA |
| MLE R | b1[B] | confidence | 0 | NA | NA |
| MLE R | b2[A] | confidence | 0 | NA | NA |
| MLE R | b2[B] | confidence | 0 | NA | NA |

## violation_late

| Method | Parameter | n paired | AB | MSE |
|---|---|---:|---:|---:|
| MLE U | a | 20 | 0.091063063 | 0.069375833 |
| MLE U | b1[A] | 20 | 0.38101293 | 1.0007585 |
| MLE U | b2[A] | 20 | 0.052365938 | 0.064089337 |
| MLE U | b1[B] | 20 | 0.76601069 | 3.0892274 |
| MLE U | b2[B] | 20 | 0.17432059 | 0.34880194 |
| MLE R | a | 20 | 0.091063063 | 0.069375833 |
| MLE R | b1[A] | 20 | 0.28331056 | 0.39617975 |
| MLE R | b2[A] | 20 | 0.71668944 | 0.82955862 |
| MLE R | b1[B] | 20 | 0.27983436 | 1.0612926 |
| MLE R | b2[B] | 20 | 1.2201656 | 2.4717895 |
| Independent U | a | 20 | 0.082042421 | 0.067366088 |
| Independent U | b1[A] | 20 | 0.40367045 | 1.069561 |
| Independent U | b2[A] | 20 | 0.058222104 | 0.068384968 |
| Independent U | b1[B] | 20 | 0.84116897 | 3.4621348 |
| Independent U | b2[B] | 20 | 0.1988575 | 0.39450343 |
| Independent R | a | 20 | 0.08110367 | 0.067297982 |
| Independent R | b1[A] | 20 | 0.33479164 | 0.42163351 |
| Independent R | b2[A] | 20 | 0.79662892 | 1.004535 |
| Independent R | b1[B] | 20 | 0.34304788 | 1.1002876 |
| Independent R | b2[B] | 20 | 1.3803306 | 3.1208982 |
| Hierarchical U | a | 20 | 0.060654621 | 0.046077865 |
| Hierarchical U | b1[A] | 20 | 0.35182051 | 0.81135225 |
| Hierarchical U | b2[A] | 20 | 0.078591717 | 0.057144817 |
| Hierarchical U | b1[B] | 20 | 0.56720975 | 1.7655446 |
| Hierarchical U | b2[B] | 20 | 0.14645085 | 0.17975731 |
| Hierarchical R | a | 20 | 0.051623407 | 0.043810275 |
| Hierarchical R | b1[A] | 20 | 0.36296642 | 0.35068325 |
| Hierarchical R | b2[A] | 20 | 0.76446955 | 0.84410728 |
| Hierarchical R | b1[B] | 20 | 0.52570154 | 0.72205272 |
| Hierarchical R | b2[B] | 20 | 1.1750223 | 1.9243434 |

| Method | Parameter | Interval | n | CP | AW |
|---|---|---|---:|---:|---:|
| Hierarchical U | a | credible | 20 | 0.9 | 0.78221195 |
| Hierarchical U | b1[A] | credible | 20 | 0.8 | 2.0456991 |
| Hierarchical U | b1[B] | credible | 20 | 0.95 | 3.4053212 |
| Hierarchical U | b2[A] | credible | 20 | 1 | 0.87425695 |
| Hierarchical U | b2[B] | credible | 20 | 1 | 1.4040111 |
| Hierarchical R | a | credible | 20 | 0.9 | 0.77798312 |
| Hierarchical R | b1[A] | credible | 20 | 0.5 | 1.1873893 |
| Hierarchical R | b1[B] | credible | 20 | 0.55 | 1.9770026 |
| Hierarchical R | b2[A] | credible | 20 | 0.25 | 1.3156539 |
| Hierarchical R | b2[B] | credible | 20 | 0.45 | 2.1988817 |
| Independent U | a | credible | 20 | 0.9 | 0.85108809 |
| Independent U | b1[A] | credible | 20 | 0.75 | 2.2307518 |
| Independent U | b1[B] | credible | 20 | 0.9 | 4.0851194 |
| Independent U | b2[A] | credible | 20 | 0.9 | 0.9108667 |
| Independent U | b2[B] | credible | 20 | 1 | 1.6292914 |
| Independent R | a | credible | 20 | 0.9 | 0.84685522 |
| Independent R | b1[A] | credible | 20 | 0.5 | 1.3153053 |
| Independent R | b1[B] | credible | 20 | 0.6 | 2.4635166 |
| Independent R | b2[A] | credible | 20 | 0.35 | 1.4576564 |
| Independent R | b2[B] | credible | 20 | 0.45 | 2.7381972 |
| MLE U | a | confidence | 20 | 0.85 | 0.86390822 |
| MLE U | b1[A] | confidence | 20 | 0.75 | 2.4155835 |
| MLE U | b1[B] | confidence | 20 | 0.8 | 4.490715 |
| MLE U | b2[A] | confidence | 20 | 1 | 1.0850195 |
| MLE U | b2[B] | confidence | 20 | 0.9 | 1.955995 |
| MLE R | a | confidence | 20 | 0.85 | 0.86390822 |
| MLE R | b1[A] | confidence | 0 | NA | NA |
| MLE R | b1[B] | confidence | 0 | NA | NA |
| MLE R | b2[A] | confidence | 0 | NA | NA |
| MLE R | b2[B] | confidence | 0 | NA | NA |

## Interpretation and limits

The unrestricted hierarchy has lower observed MSE than unrestricted independent Bayes for every primary parameter in all three scenarios. Its shape MSE is 0.036386, 0.028812 and 0.046078 under strict ordering, equality and violation, versus MLE 0.051452, 0.037730 and 0.069376. At equality, restriction improves cause 1 and worsens cause 2 in both blocks for all three estimator families. Under false ordering, restricted Bayesian cause-2 coverage is 0.25--0.45; nominal posterior probability does not protect against a false model constraint.

The order constraint leaves classical a and B unchanged. Under the independent scale prior, it also leaves their posterior distributions unchanged; separate method seeds introduce small Monte Carlo differences. This invariance does not hold for the hierarchy. Under equality, pooling and posterior restriction trade variance against asymmetric bias; inspect both causes and blocks. Under violation, any accuracy gain for one coefficient does not validate the false ordering: the other coefficient can deteriorate substantially.

All ranks are descriptive, by scenario and parameter, on identical full-six dataset IDs. No aggregate ranking across differently scaled parameters is used. There are only 20 independent replications per scenario; CP changes in steps of 0.05, with binomial SE about 0.049 at CP=0.95. MSE difference Monte Carlo errors are supplied in relative_mse_comparisons.csv. Posterior mean MCSE is recorded parameter by parameter; ESS and Rhat do not prove convergence.

Classical log-Wald intervals respect positivity. Raw Wald intervals are retained with their original labels and negative endpoints flagged in issues.csv. Restricted individual-b confidence intervals are unsupported and remain unavailable. Exact restricted probability sets can be empty; these count as zero coverage and zero width. Bayesian credible intervals are shortest contiguous sample intervals, not guaranteed disconnected HPD regions. Coverage is a frequentist simulation property even for credible intervals.

Runtime summaries mix historical and optimized implementations and baseline/extended chain lengths, so they describe actual cost rather than a controlled sampler speed comparison. All-attempt totals include preserved failed fits. Posterior priors were retained without sensitivity selection; wider design and prior-sensitivity studies remain outside this 60-dataset completion.

## Reproduction

From the repository root:

```sh
.venv/bin/python -m experiments.completion_study analyze
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.completion_study fit --workers 3
```

The first command rebuilds summaries/figures from saved results only. The second resumes only unresolved fits. Final results are under results/simulation_completion_v1/latest. Historical submission summaries remain unchanged.
