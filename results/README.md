
## Results Data

Full MVPA results are available in the `results/` folder as CSV files:

- `mvpa_happiness_anger.csv`
- `mvpa_anxiety_sadness.csv`

These files contain all decoding results across tasks, masks, and conditions.

## Max-statistic correction results 

Full max statistic result for each permutation (100 perms) and the histogram (image.png) of each voxel (count) and accuracy. 

Across 100 permutations, the maximum accuracies ranged from 0.868 to 1.000, with a mean maximum accuracy of 0.932. The significance threshold was defined as the 95th percentile of this distribution (accuracy = 0.974), corresponding to a family-wise error corrected significance level of α = 0.05. The real searchlight map reached a maximum accuracy of 1.000, exceeding the corrected threshold, and resulted in 534 voxels surviving correction. Voxels exceeding this threshold were therefore considered statistically significant. These results indicate that, despite high accuracies occasionally occurring by chance in permutation maps, several regions in the real data showed decoding performance exceeding the strongest effects expected under the null distribution, suggesting robust above-chance classification after strict correction for multiple comparisons.


