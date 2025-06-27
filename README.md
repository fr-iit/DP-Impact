# The Price of Privacy: A Comprehensive Study on the Impacts of Pure Differential Privacy over Recommendation Performance, Fairness and Stability

## Abstract

Collaborative filtering-based recommender systems typically rely on large volumes of user behavioral data to provide accurate personalized recommendations. However, collecting and using such data can expose users to significant privacy risks. To mitigate these risks, Differential Privacy (DP) based privacy-preserving techniques are widely applied in recommender systems, as it ensures the required level of privacy without excessive communication and computational cost. DP can safeguard user privacy by introducing random noise, although this results in an acceptable cost of accuracy. DP-based privacy preservation can be applied at different stages of the recommendation pipeline, such as the input, objective, and output stages. However, there is limited research on how the choice of perturbation stage affects personalized recommendations and how DP influences recommendation fairness and stability. This study addresses these gaps by systematically analyzing the impact of different DP perturbation approaches on recommendation accuracy, fairness (across cold-start vs. regular users and popular vs. long-tail items), and stability. For the experiments, we applied six perturbation approaches to the Matrix Factorization-based recommender system, evaluating them on four widely-used datasets representing different sparsity levels: MovieLens 100K, MovieLens 1M, Yahoo!Movie and Amazon Beauty. Our key findings are three-fold: first, among the six perturbation schemes, input-level DP injects the smallest loss in accuracy and user fairness, whereas objective-level DP achieves the greatest gains in stability and item fairness. Second, DP reduces popularity bias, with the effect becoming more pronounced under stronger privacy settings (lower privacy budgets). Finally, recommendations become more stable when DP perturbation is applied at the objective stage, revealing a new trade-off between privacy, accuracy, and stability in DP-protected recommender systems. 

## Requirements
  * Python 3
  * Sklearn
  * Numpy
  * Pandas
  * Matplotlib

## Instruction

  * You must have the MovieLens (https://grouplens.org/datasets/movielens/), yahoo movie (https://webscope.sandbox.yahoo.com/) and Amazon All_Beauty data (https://amazon-reviews-2023.github.io/) downloaded in your project.
  * Use 1. 'ml100k' ; 2. 'ml1m' ; 3. 'yahoo' ; 4. 'ml10m'; 5. 'beauty' keywords as the value of dataset_name variable to load respective datset through DataLoader.py
  * Use the following keyword to execute the respective perturbation method for analysis
    1. 'NoDP' -> denotes non-privacy-preserved matrix factorization recommender system
    2. 'UDP-MF' -> Unbound input perturbation approach
    3. 'BDP-MF' -> Bound input perturbation approach
    4. 'DP-SGD' ->  Stochastic gradient perturbation (objective perturbation)
    5. 'VP-DPMF' -> Vector Perturbation (objective perturbation)
    6. 'DP-LMMF' -> Latent Factor Perturbation (output perturbation)
    7. 'DP-PRMF' -> Predicted Rating Perturbation (output perturbation)
  * Use epsilion = {0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0}
