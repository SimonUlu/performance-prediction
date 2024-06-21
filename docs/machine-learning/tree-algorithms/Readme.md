# This doc should give you a brief overview on how we implemented our machine learning models

Experimenting with these hyperparameters can help you find a more accurate and generalized model for your specific dataset. It's common practice to use cross-validation and hyperparameter tuning techniques (such as grid search or random search) to determine the best settings for your problem. Remember, the goal is to find a good balance between model complexity (to accurately capture the underlying patterns) and model generalization (to perform well on unseen data).

## 1. Decison-Tree Regressor



## 2. Random Forest Regressor

- Most important hyperparameters play with them

##### a. n_estimators

What it does: Indicates the number of trees in the forest.
Default value: 100
To note: A higher number of trees increases the prediction accuracy, but requires more computing time and can lead to a point of diminishing returns.

##### b. max_depth

What it does: The maximum depth of the trees.
Default value: None (trees are expanded until all leaves are pure or contain less than min_samples_split samples).
To note: Limiting the depth can help avoid overfitting.

##### c. min_samples_split

What it does: The minimum number of samples required to split a node.Default value: 2To note:Higher values can help prevent overfitting, but if they are too high, the model may underfit.

##### d. max_features

What it does: The number of features to consider when searching for the best split.Default value: 1.0 (which means that all features are taken into account)To note:Selecting a smaller number of features can increase the diversity between the trees and lead to a more robust model.

##### e. bootstrap

What it does: Whether bootstrap samples are used when creating trees.
Default value: TrueTo 
note:Bootstrap samples (with backspacing) help reduce variance, but it may be interesting to set this to False to see how this affects your specific problem.

## 3. Gradient Boosting Regressor

##### a. n_estimators

What it does: Specifies the number of boosting stages the model will be built.
Default value: 100
Considerations: Increasing the number of stages can improve the model's accuracy but also increases the risk of overfitting and computational cost. There's a trade-off to find the optimal number.

##### b. learning_rate

What it does: Shrinks the contribution of each tree by learning_rate.
Default value: 0.1
Considerations: There's a trade-off between learning_rate and n_estimators. Lower learning_rate values require more trees to model all the relations properly, but can lead to better generalization.

##### c. max_depth
What it does: Maximum depth of the individual regression estimators.
Default value: 3
Considerations: Controls the depth of the tree. Deeper trees can model more complex relations but also can lead to 
overfitting. Adjusting this parameter can help balance model complexity and generalization.


##### d. min_samples_split
What it does: The minimum number of samples required to split an internal node.
Default value: 2
Considerations: Higher values prevent creating nodes that contain too few samples, potentially reducing overfitting, but could lead to underfitting if set too high.

##### e. min_samples_leaf

What it does: The minimum number of samples required to be at a leaf node.
Default value: 1
Considerations: Similar to min_samples_split, setting this higher can help in preventing overfitting.

##### f. max_features

What it does: The number of features to consider when looking for the best split.
Default value: None (which means max_features=n_features)
Considerations: Reducing max_features can increase the model's robustness to noise by creating more diversified trees, but too low values can lead to underfitting.

##### g. subsample

What it does: The fraction of samples to be used for fitting the individual base learners.
Default value: 1.0
Considerations: Using a value less than 1.0 can lead to a reduction of variance and an increase in bias. It enables the stochastic gradient boosting feature.

##### h. random_state

What it does: Controls the randomness of the bootstrapping of the samples used when building trees (if subsample < 1.0) and the sampling of the features to consider when looking for the best split at each node.
Considerations: Setting a random state ensures your results are reproducible.