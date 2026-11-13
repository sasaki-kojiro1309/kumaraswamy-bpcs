# runtime summary

CSV is authoritative at full round-trip precision. R columns are the explicit metric denominators. MSE ranks are descriptive and scenario/parameter specific; only 20 independent replications per scenario.

| method | selected_fits | mean_selected_seconds | median_selected_seconds | total_all_attempt_seconds |
| --- | --- | --- | --- | --- |
| mle_unrestricted | 60 | 0.010537857617 | 0.0092270210007 | 0.63227145702 |
| mle_within_block | 60 | 0.00989653398343 | 0.00858360400161 | 0.593792039006 |
| bayes_independent_unrestricted | 60 | 12.6494692986 | 10.6130647915 | 758.968157917 |
| bayes_independent_within_block | 60 | 12.4732931048 | 10.650338979 | 748.397586291 |
| bayes_hierarchical_unrestricted | 60 | 75.0174670966 | 80.791244542 | 4513.55854025 |
| bayes_hierarchical_within_block | 60 | 80.0591019556 | 56.967736229 | 6421.92771238 |
