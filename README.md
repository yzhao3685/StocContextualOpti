# Estimate-Then-Optimize Versus Integrated-Estimation-Optimization: A Stochastic Dominance Perspective

This repository contains code pertaining to the paper "Estimate-Then-Optimize Versus Integrated-Estimation-Optimization: A Stochastic Dominance Perspective
" by A. Elmachtoub, H. Lam, H. Zhang and Y. Zhao. Arxiv link: [https://arxiv.org/abs/2302.12736](https://arxiv.org/abs/2304.06833)

In data-driven stochastic optimization, model parameters of the underlying distribution need to be estimated from data in addition to the optimization task. Recent literature considers integrating the estimation and optimization processes by selecting model parameters that lead to the best empirical objective performance. This integrated approach, which we call integrated-estimation-optimization (IEO), is known to outperform simple estimate-then-optimize (ETO) when the model is misspecified. In this paper, we show that a reverse behavior appears when the model class is well-specified and there is sufficient data. Specifically, for a general class of nonlinear stochastic optimization problems, we show that simple ETO outperforms IEO when the model class covers the ground truth, in the strong sense of stochastic dominance of the asymptotic regret. Namely, the entire distribution of the asymptotic regret, not only its mean or other moments, is always better for ETO compared to IEO. Our results also apply to constrained, contextual optimization problems where the decision depends on observed features. Whenever applicable, we also demonstrate how standard empirical optimization (EO) performs the worst when the model class is well-specified in terms of asymptotic regret, and best when it is misspecified. Finally, we provide experimental results to support our theoretical comparisons and illustrate when our insights hold in finite-sample regimes and under various degrees of misspecification.


## Installation

Install the required packages listed below and the latest version of Gurobi optimization solver.

```
pip install sklearn
pip install scipy
pip install matplotlib 
```

## Running Experiments

Use files in the 'newsvendor' folder to reproduce newsvendor experiments.
Use files in the 'portfolio_optimization' folder to reproduce portfolio optimization experiments.


