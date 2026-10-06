# Starbucks Promotion Optimization --- Uplift & Time-Series Modeling

## Overview

This project develops a data-driven promotion targeting strategy using
the Starbucks Customer Rewards dataset.

The goal is not simply to predict which customers are likely to
purchase. Instead, the project focuses on identifying customers whose
spending is expected to **increase because they viewed a promotional
offer**.

The workflow combines:

-   Event-log reconstruction
-   Accidental completion handling
-   Offer attribution
-   Time-series feature engineering
-   T-Learner and S-Learner uplift modeling
-   Qini and IPW evaluation
-   Robustness and placebo checks
-   \$2,000 budget-constrained optimization

The overall business question is:

> **Who is likely to buy more because of the offer?**

------------------------------------------------------------------------

## Business Objective

Promotional offers such as discounts and BOGOs can reduce margins when
they are sent to customers who would have purchased anyway.

The objective is therefore to identify **Persuadable customers** ---
customers whose purchasing behavior is positively changed by promotional
exposure --- and allocate a fixed **\$2,000 promotional budget** toward
the customers with the highest expected incremental value.

------------------------------------------------------------------------

## Dataset

The Starbucks Customer Rewards dataset contains three main files:

-   `portfolio.json` --- offer metadata such as offer type, reward,
    difficulty, and duration.
-   `profile.json` --- customer demographic information.
-   `transcript.json` --- time-series events including offer received,
    offer viewed, offer completed, and transactions.

The raw event log is reconstructed into customer-offer instances to
analyze the complete offer journey.

------------------------------------------------------------------------

# Project Workflow

## Phase 1 --- Event Log Wrangling

The first phase reconstructs the customer journey from the raw event
log.

### Key tasks

-   Match offer views and completions to the correct offer receipt.
-   Handle customers who receive the same offer multiple times.
-   Calculate spending within offer windows.
-   Identify accidental completions.
-   Attribute transactions to viewed offers.
-   Handle overlapping offer periods.

------------------------------------------------------------------------

## Accidental Completion Trap

An important issue in the event log is that some customers completed an
offer without a recorded offer-view event beforehand.

These observations were classified as **accidental completions**.

Three treatment definitions were evaluated:

1.  Treat accidental completers as control.
2.  Exclude accidental completers.
3.  Treat accidental completers as treated.

The primary analysis treats accidental completers as **control** because
treatment is defined around actually viewing the promotional offer.

### Observed groups

  Group                         Observations   Mean Spend
  --------------------------- -------------- ------------
  Accidental completers                9,834      \$52.35
  Never viewed                         8,519       \$4.36
  Normally viewed/completed           42,689      \$31.04

This distinction prevents offer completion from being incorrectly
interpreted as evidence that the promotion caused the customer's
spending behavior.

------------------------------------------------------------------------

# Phase 2 --- Time-Series Feature Engineering

The project creates behavioral features using only information available
**before the offer is received**.

This prevents future-data leakage.

## 1. Offer Fatigue

Measures the number of offers received in trailing:

-   14 days
-   30 days

It captures recent promotional exposure and possible offer saturation.

## 2. Inter-Purchase Time (IPT)

Measures the average time between organic, non-offer transactions.

It captures the customer's natural purchasing rhythm.

## 3. Reward Hunting Index

Measures the relationship between spending during active viewed-offer
periods and spending during organic periods.

It provides a behavioral indication of promotional responsiveness and
price sensitivity.

## 4. Time-Decayed AOV

Calculates average order value while assigning greater importance to
recent purchases.

A **7-day half-life** was used, meaning a transaction seven days old
receives approximately half the weight of a current transaction.

A sensitivity analysis was also performed using 3-, 7-, and 14-day
half-lives.

------------------------------------------------------------------------

## Leakage Test

A dedicated leakage test was performed on 400 randomly selected
customer-offer snapshots.

Future transactions, offer receipts, and offer views were removed before
rebuilding the features.

Results:

  Test                      Result
  ----------------------- --------
  Customer-waves tested        400
  Mismatches                     0
  Largest difference             0

This provides evidence that the tested feature-engineering pipeline uses
information available before offer receipt.

------------------------------------------------------------------------

# Phase 3 --- Uplift Modeling

The project uses **Conditional Average Treatment Effect (CATE)**
modeling rather than a standard purchase-probability classifier.

The objective is to estimate:

> How much additional spending is expected because of the treatment?

## T-Learner

Two separate outcome models are trained:

-   One model for treated customers
-   One model for control customers

Uplift is estimated as:

`Predicted Spend if Treated - Predicted Spend if Control`

## S-Learner

A single model is trained using treatment as an additional input.

Each customer is scored under:

-   Treatment = 1
-   Treatment = 0

The difference between the two predictions is the estimated uplift.

## Base Model

`HistGradientBoostingRegressor` is used because the outcome is
continuous spending and the model can capture nonlinear relationships
and interactions.

------------------------------------------------------------------------

# Target Definition

Treatment is based on viewing the promotional offer within the defined
treatment window.

The modeled outcome is customer spending during the:

**24--72 hour period after offer receipt**

This makes the target a continuous revenue outcome rather than a binary
purchase indicator.

------------------------------------------------------------------------

# Time-Based Validation

Random train-test splitting was avoided because future observations
should not influence historical model training.

Forward-chaining validation was used:

``` text
Past observations → Future validation observations
```

### Cross-validation results

  Model       Selected Configuration     Mean Qini
  ----------- ------------------------ -----------
  T-Learner   Shallow                    **0.762**
  S-Learner   Deep Slow                  **0.653**

The T-Learner was therefore the stronger model during forward
cross-validation.

------------------------------------------------------------------------

# Phase 4 --- Model Evaluation

Standard classification metrics such as accuracy and F1-score are not
the main evaluation metrics because this is an uplift problem.

The main evaluation metric is the **Qini coefficient**, which evaluates
how effectively the model ranks customers by incremental response
compared with random targeting.

## Final Holdout Results

  Model              Qini    IPW Qini   Mean Predicted Uplift
  ----------- ----------- ----------- -----------------------
  T-Learner     **1.036**       0.364                 \$3.203
  S-Learner         1.013   **0.381**                 \$3.352

Both models produced positive Qini values.

### Bootstrap Qini Confidence Intervals

-   T-Learner: **\[0.911, 1.162\]**
-   S-Learner: **\[0.892, 1.127\]**

### IPW Qini Confidence Intervals

-   T-Learner: **\[0.185, 0.518\]**
-   S-Learner: **\[0.204, 0.542\]**

------------------------------------------------------------------------

# Inverse Propensity Weighting

Offer viewing was not randomized, so treated and control customers can
have different characteristics.

A propensity model estimates:

`P(Viewing Offer | Customer Features)`

Inverse Propensity Weighting (IPW) is then used as an additional
evaluation adjustment for observed treatment-selection differences.

Propensity scores were clipped to:

`0.05 ≤ propensity ≤ 0.95`

This prevents extremely large IPW weights.

IPW reduces bias from observed treatment-selection differences but does
**not** by itself prove causality or eliminate unobserved confounding.

------------------------------------------------------------------------

# Robustness Checks

The project includes several additional checks:

-   Bootstrap confidence intervals
-   Capped vs. uncapped outcome sensitivity
-   Propensity overlap diagnostics
-   Negative-uplift analysis
-   Placebo treatment-label test
-   Treatment-definition sensitivity
-   Time-decay sensitivity
-   Feature leakage testing

The placebo test randomly shuffles treatment labels within each time
wave and retrains the models. Real model performance was substantially
stronger than the placebo results.

------------------------------------------------------------------------

# Budget Optimization

The final stage converts predicted uplift into an actionable marketing
strategy.

The available promotional budget is:

**\$2,000**

Net Incremental Revenue is defined as:

`NIR = Incremental Revenue - Promotion Cost`

Three allocation approaches were compared.

### 1. Prefix

Rank customers by predicted uplift and select the highest-ranked
customers until the budget is reached.

### 2. NIR per Dollar

Prioritize customers according to predicted net incremental value per
dollar of promotion cost.

### 3. Exact Optimization

Use a dynamic-programming / knapsack-style optimizer to find the best
feasible customer combination under the budget.

The exact optimizer was tested against exhaustive brute-force solutions
on 200 random toy problems and produced **zero mismatches**.

------------------------------------------------------------------------

# Final Business Results

  --------------------------------------------------------------------------------
  Model +            Customers   Predicted NIR    Observed NIR             IPW NIR
  Strategy                                                     
  ------------- -------------- --------------- --------------- -------------------
  T-Learner                470       \$3,102.5       \$4,626.5          $3,221.7 |
  Prefix                                                         | T-Learner NIR/$

  T-Learner                977       \$5,715.9       \$6,373.5           \$5,106.2
  Exact                                                        

  S-Learner                429       \$2,322.7       \$3,479.4          $1,988.8 |
  Prefix                                                         | S-Learner NIR/$

  **S-Learner          **971**   **\$4,883.2**   **\$7,173.5**       **\$6,240.3**
  Exact**                                                      
  --------------------------------------------------------------------------------

## Final Strategy

The strongest final business allocation was:

**S-Learner + Exact Optimization**

Results:

-   Customers targeted: **971**
-   Budget: approximately **\$2,000**
-   Predicted NIR: **\$4,883.2**
-   Observed NIR: **\$7,173.5**
-   IPW-adjusted NIR: **\$6,240.3**
-   IPW 95% CI: **\[\$3,906, \$8,807\]**

The T-Learner was the strongest model during forward validation, while
the S-Learner with exact budget optimization produced the strongest
final business outcome.

------------------------------------------------------------------------

# Key Takeaway

A customer being likely to purchase does not automatically mean that
customer should receive a promotion.

The more useful question is:

> **Will the customer purchase more because of the promotion?**

This project addresses that question using behavioral time-series
features, uplift modeling, causal evaluation diagnostics, and budget
optimization.

The final strategy demonstrates how a limited promotional budget can be
directed toward customers with higher expected incremental value instead
of distributing offers broadly.

------------------------------------------------------------------------

# Technologies Used

-   Python
-   pandas
-   NumPy
-   scikit-learn
-   Matplotlib
-   Seaborn
-   Jupyter Notebook

------------------------------------------------------------------------

# Suggested Repository Structure

``` text
Starbucks-Promotion-Uplift-Optimization/
│
├── README.md
├── notebooks/
│   └── Starbucks_Uplift_Modeling.ipynb
│
├── data/
│   ├── portfolio.json
│   ├── profile.json
│   └── transcript.json
│
├── outputs/
│   ├── phase4_selected_customers.csv
│   ├── phase4_budget_summary.csv
│   └── phase4_qini_summary.csv
│
├── figures/
│
├── requirements.txt
└── .gitignore
```

------------------------------------------------------------------------

# Author

**Sujal Bhavsar**



------------------------------------------------------------------------

## Disclaimer

This project is an analytical modeling exercise based on the provided
Starbucks Customer Rewards dataset. The uplift and revenue estimates are
model-based and should not be interpreted as guaranteed future business
results. IPW adjusts for observed treatment-selection differences but
does not establish causality in the presence of unobserved confounding.
