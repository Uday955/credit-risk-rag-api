# Fairness Audit Report — Credit Risk Prediction Model

## 1. Objective & Scope

This audit evaluates whether the credit default prediction model treats 
different groups of applicants fairly, in line with regulatory expectations 
such as the Equal Credit Opportunity Act (ECOA).

### Limitation: No direct protected-class data available
The dataset used (LendingClub, 2007-2018) does not include race, gender, or age — 
common in public lending datasets, partly due to regulations restricting the use 
of such attributes in underwriting decisions. As a result, this audit uses 
**`addr_state` (applicant's US state)** as a geographic proxy attribute, since 
geography can correlate with protected characteristics (e.g., through historical 
housing patterns).

This is a best-effort proxy audit, consistent with the data constraints many 
real-world lenders face when auditing models built on anonymized or regulation-compliant 
datasets. A production-grade audit would ideally test actual protected attributes 
directly, collected specifically for fairness monitoring purposes, separate from 
the model's input features.

### States analyzed
Analysis focused on the 5 states with the largest sample sizes in the test set, 
to ensure statistically meaningful comparisons:
- California (CA): 39,341 applicants
- Texas (TX): 21,993 applicants
- New York (NY): 21,864 applicants
- Florida (FL): 19,254 applicants
- Illinois (IL): 10,359 applicants


-----------------------------------------------------------------------------------------------------------------------------------------------


## 2. Methodology

Three complementary fairness metrics were used, moving from a simple surface-level 
check to a more rigorous analysis:

### 2.1 Selection Rate / Demographic Parity
Measures the percentage of applicants predicted as high-risk (default = 1) within 
each state. A large difference between states' selection rates is a first-pass 
signal worth investigating further.

### 2.2 Disparate Impact Ratio
A standard regulatory benchmark (the "80% rule" / four-fifths rule used by US 
regulators): the ratio of the lowest group's selection rate to the highest group's 
selection rate. A ratio below 0.80 is typically flagged as a potential fairness 
concern requiring further investigation.

### 2.3 Equalized Odds (True Positive Rate & False Positive Rate by group)
Demographic parity alone does not account for genuine differences in underlying 
risk between groups. Equalized odds instead examines the model's **errors**:

- **True Positive Rate**: Of applicants who actually defaulted, what fraction did 
  the model correctly flag, per state?
- **False Positive Rate**: Of applicants who actually repaid, what fraction did 
  the model incorrectly flag as risky, per state?

If these error rates are similar across states, observed differences in raw 
selection rates likely reflect genuine differences in applicant risk profiles 
rather than model unfairness. Significant differences in error rates are a 
stronger signal of a genuine fairness issue.

### Tooling
All fairness metrics were calculated using **Fairlearn**, an open-source toolkit 
developed by Microsoft for assessing and mitigating unfairness in ML models.


-----------------------------------------------------------------------------------------------------------------------------------------------


## 3. Findings

### 3.1 Selection Rate by State

| State | Selection Rate (flagged as high-risk) |
|-------|----------------------------------------|
| IL    | 2.92%                                   |
| TX    | 3.40%                                   |
| CA    | 3.65%                                   |
| FL    | 3.94%                                   |
| NY    | 4.24%                                   |

**Disparate Impact Ratio (IL vs NY): 0.688**

This falls below the 0.80 regulatory benchmark, indicating a potential fairness 
concern between the lowest-flagged group (Illinois) and the highest-flagged group 
(New York) that warrants deeper investigation.

### 3.2 Equalized Odds by State

| State | True Positive Rate | False Positive Rate |
|-------|--------------------:|----------------------:|
| IL    | 0.092               | 0.015                 |
| TX    | 0.093               | 0.019                 |
| CA    | 0.111               | 0.018                 |
| FL    | 0.103               | 0.021                 |
| NY    | 0.114               | 0.022                 |

**Key observation:** New York shows both the highest true positive rate AND the 
highest false positive rate, while Illinois shows the lowest of both. This is a 
systematic pattern, not random noise — the model is consistently more "aggressive" 
in flagging New York applicants as risky (both correctly and incorrectly) and more 
"lenient" toward Illinois applicants.

### Interpretation
This pattern is consistent with two possible explanations, which this analysis 
cannot fully distinguish between given available data:
1. **Genuine risk differences** — New York's loan applicant population may carry 
   real, legitimate differences in average credit risk compared to Illinois.
2. **Model bias amplification** — the model may be over-weighting state-level 
   signals (directly or via correlated features) beyond what true risk differences 
   justify.

Both the magnitude of the disparity and its systematic (non-random) nature justify 
treating this as a genuine finding requiring mitigation and ongoing monitoring, 
rather than dismissing it as noise.


-----------------------------------------------------------------------------------------------------------------------------------------------


## 4. Mitigation Attempt

### Technique: Post-processing via ThresholdOptimizer (Fairlearn)

A post-processing mitigation approach was selected over pre-processing or 
in-processing alternatives, since it allows adjustment of decision thresholds 
per group without retraining or modifying the already-validated, calibrated 
model. The `equalized_odds` constraint was used, directly targeting the 
disparity identified in Section 3.2.

### Results: Fairness metrics after mitigation

| State | True Positive Rate (after) | False Positive Rate (after) |
|-------|------------------------------:|-------------------------------:|
| IL    | 0.073                        | 0.017                         |
| TX    | 0.072                        | 0.015                         |
| CA    | 0.072                        | 0.015                         |
| FL    | 0.071                        | 0.015                         |
| NY    | 0.073                        | 0.015                         |

The disparity across states was successfully and substantially reduced — true 
positive rates converged to a tight 0.071-0.073 range (down from 0.092-0.114), 
and false positive rates converged to 0.015-0.017 (down from 0.015-0.022).

### Critical finding: Unacceptable cost to model utility

| Metric (overall) | Original Model | After Mitigation |
|-------------------|------------------:|---------------------:|
| Recall (default class) | 0.68 | **0.07** |
| Precision (default class) | 0.33 | 0.55 |

While fairness metrics improved, **overall recall for detecting actual defaults 
collapsed from 68% to 7%**. Investigation revealed the ThresholdOptimizer achieved 
equalized error rates primarily by raising decision thresholds dramatically across 
*all* groups, making the model rarely predict "default" for anyone, regardless of 
state. This satisfies the mathematical fairness constraint but renders the model 
practically useless for its core purpose — identifying risky loans.

### Decision: Mitigation rejected

This is a well-documented failure mode in fairness-aware machine learning: a model 
can achieve statistical fairness by becoming uniformly unhelpful, rather than by 
becoming genuinely more equitable. Given this trade-off, **the mitigated model was 
rejected**, and the original (unmitigated) calibrated model was retained as the 
project's final model.

This finding is itself a valuable outcome: it demonstrates that fairness metrics 
must always be evaluated alongside business/utility metrics, not in isolation, 
and that applying a fairness technique is not inherently beneficial without 
verifying its full impact.


-----------------------------------------------------------------------------------------------------------------------------------------------


## 5. Recommendations & Next Steps

Based on this audit, the following recommendations are made:

1. **Retain the original (unmitigated) model** as the production candidate, given 
   that the tested mitigation technique traded away unacceptable amounts of 
   predictive utility for fairness gains.

2. **Investigate the root cause of the state-level disparity further**, specifically:
   - Compare the actual historical default rates of New York vs. Illinois 
     applicants in the raw data, independent of model predictions, to assess 
     whether the disparity reflects genuine population-level risk differences.
   - Examine whether `addr_state`'s influence in the model (via SHAP, see 
     Week 4 analysis) is proportionate, or whether the model may be over-relying 
     on geography relative to more direct risk indicators (e.g., credit score, 
     income, debt-to-income ratio).

3. **Explore alternative mitigation strategies** in future iterations, such as:
   - In-processing techniques (e.g., Fairlearn's `ExponentiatedGradient`) that 
     incorporate fairness constraints directly during model training, which may 
     achieve a better fairness-utility trade-off than post-processing.
   - Reducing or re-weighting the model's reliance on `addr_state` as a feature, 
     and re-evaluating both fairness and performance metrics.

4. **Expand the audit when richer data is available.** If protected-class data 
   (e.g., age, verified demographic information collected specifically for 
   fairness monitoring, under appropriate legal frameworks) becomes available, 
   a more direct fairness audit should be conducted rather than relying on 
   geographic proxies alone.

5. **Establish ongoing monitoring**, not a one-time check. Fairness properties 
   of a deployed model can drift over time as applicant populations and economic 
   conditions change; this audit should be re-run periodically (e.g., quarterly) 
   rather than treated as a single point-in-time certification.

---

## Summary

This audit identified a statistically meaningful disparity in how the model treats 
applicants across US states, used as a proxy given the absence of direct protected-
class data. A standard mitigation technique successfully reduced this disparity 
but at an unacceptable cost to the model's core predictive utility, and was 
therefore rejected. The original model is retained, with clear, documented 
recommendations for further investigation and monitoring.