# A preliminary study on the equivalence between deterministic and probabilistic methods

Abstract

This study explores the equivalence between deterministic and probabilistic methods in the context of chess artificial intelligence (AI), focusing on two AI models: Kane and Abel. Kane represents the deterministic approach, while Abel embodies the probabilistic methodology. Leveraging Avi Wigderson's theories on computational complexity, randomization, and derandomization, we examine the conditions under which these two seemingly distinct methods converge. We apply Kolmogorov-Arnold Networks (KANs) to analyze the equivalence, providing symbolic representations that illustrate the underlying functional similarities between deterministic and probabilistic algorithms. Through a series of experiments, including heuristic adjustments and transformations, we demonstrate that with increasing iterations, both methods can achieve similar outcomes, validated by convergence in equivalence score curves and algorithmic equilibrium. The findings contribute to the broader understanding of AI development, particularly in optimizing decision-making processes in complex environments like chess. Future research will delve deeper into the structural aspects of equivalence, symmetry applications, further enhancing the theoretical and practical implications of this study.

File Specification:
1. Basic-algorithms-Kane-Abel_implementation.ipynb: This file includes the implementation of Kane's and Abel's basic algorithms:

1.1 Kane: Rule-Based Chess AI

1.1.1 Basic Deterministic Algorithms:

The design of Kane's basic algorithms includes two main components: Minimax and Alpha-Beta Pruning. The Minimax algorithm evaluates nodes by alternating between maximizing and minimizing players to find the best move. Alpha-Beta Pruning improves this process by introducing alpha and beta values to prune branches, making the search more efficient.

1.1.2 Optimized Pruning Algorithms:

Principal Variation Search (PVS): Focuses on the principal variation—the best sequence of moves—by dynamically adjusting the search window based on the expected move sequence.
Null Move Pruning: Reduces the search depth by assuming the opponent makes a null move, effectively skipping their turn.
Late Move Reductions (LMR): Reduces the search depth for less promising moves.
Aspiration Windows: Quickly cuts off the search around the expected score.
Multi-Cut: Prunes branches earlier by making multiple cuts in the search tree.
Enhanced Transposition Table Management: Improves efficiency in storing and retrieving previously computed positions.
Fail-soft Alpha-Beta: Returns bounds on the score when it fails to find an exact value.
Minimum Window Search: Uses a minimal search window to quickly determine score bounds.
MTD(n,f): A variant of Alpha-Beta that searches with minimal windows and updates bounds.

1.2  Abel: Machine Learning-Based Chess AI

1.2.1 Basic Probabilistic Algorithms:

Simple Neural Networks (NN): A basic neural network architecture for initial strategy development.
Deep Q-Networks (DQN): Combines Q-learning with deep learning to enable agents to learn optimal strategies through experience, balancing exploration and exploitation.

1.2.2 Optimized Learning Algorithms:

Deep Reinforcement Learning: Enhances the learning process by using reward-based feedback to improve the AI’s performance over time.
Genetic Algorithms: Optimizes neural networks using evolutionary techniques.
Curriculum Learning: Gradually increases the complexity of training scenarios.
Self-Play: Allows the AI to discover new strategies by playing against itself.
Policy Gradient Methods: Uses gradient ascent to optimize policy parameters.
Actor-Critic Methods: Combines value and policy learning in reinforcement learning.

1.2.3 Other Network Algorithms:

Convolutional Neural Networks (CNNs): Processes spatial information on the chessboard.
Residual Networks (ResNets): Improves training efficiency through deep residual learning.
Recurrent Neural Networks (RNNs): Models sequences of moves and strategies.
Attention Mechanisms: Focuses on important board regions.
Graph Neural Networks (GNNs): Represents the chessboard as a graph and processes it.

2. Combined_Experiments_With_Kane_Abel.ipynb

This file includes the full design and part of the running results of the experiments with KAN integration. It was a file combined with Ex_1.ipynb,Ex_2-1.ipynb,Ex_2-2.ipynb,Ex_3.ipynb,Ex_4.ipynb,Ex_5.ipynb. 

3. IntegratedExperimentsAutoRun.ipynb

This file includes codes for automated running an experiment with KAN integration

4. Minimax_Kane_vs_NN_Abel.ipynb and Pruning_Kane_vs_DQN_Abel.ipynb:

These two files includes the codes for the initialized run for the two experiments: one is using minimax algorithms for kane and simple neural network based abel; and the other is using pruning algorithms for kane and DQN algorithms for abel. Both of them haven't integrated with KAN.
