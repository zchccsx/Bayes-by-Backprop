# Bayes: Bayesian Neural Networks for Regression

This repository implements Bayesian Neural Networks (BNNs) for regression tasks using the **Bayes by Backprop** technique. BNNs provide not just point predictions but also uncertainty estimates, which are crucial for many real-world applications.

## Key Features

- **Probabilistic Weights**: Instead of single-point estimates, the network represents weights as probability distributions
- **Uncertainty Quantification**: Get confidence intervals for predictions
- **Automatic Regularization**: Weight uncertainty naturally prevents overfitting
- **Flexible Architecture**: Easy-to-customize neural network architecture
- **Feature Importance Analysis**: Analyze which features contribute most to predictive uncertainty

## Installation

First, clone this repository:

```bash
git clone https://github.com/yourusername/Bayes-by-Backprop.git
cd Bayes-by-Backprop
```

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run one of the example scripts to see BNNs in action:

```bash
# Real-world dataset example
python -m Bayes.examples.california_housing_regression

# Toy 1D regression example with visualization
python -m Bayes.examples.toy_regression
```

## Framework Structure

The implementation follows a modular structure:

```
Bayes/
├── modules/             # Bayesian neural network layers
│   ├── base_bayesian_module.py  # Base class for Bayesian modules
│   ├── linear_bayesian_layer.py # Bayesian linear layer
│   └── weight_sampler.py        # Weight sampling utilities
├── losses/              # Loss functions
│   └── kl_divergence.py         # KL divergence calculation
├── models/              # Model implementations
│   └── regression.py            # Bayesian regressor model
├── utils/               # Utilities
│   ├── variational_estimator.py # Variational inference decorator
│   ├── regression_trainer.py    # Trainer for regression models
│   └── data_utils.py            # Data processing utilities
└── examples/            # Example scripts
    ├── california_housing_regression.py  # Real-world dataset example
    └── toy_regression.py                 # 1D toy example
```

## Usage

### Creating a Model

```python
from Bayes.models import BayesianRegressor

# Create a Bayesian Neural Network
model = BayesianRegressor(
    input_dim=X_train.shape[1],  # Number of features
    hidden_dims=[64, 32],        # Hidden layer dimensions
    output_dim=1                 # Output dimension (1 for regression)
)
```

### Training the Model

```python
from Bayes.utils import BayesianRegressionTrainer

# Create trainer
trainer = BayesianRegressionTrainer(
    model=model,
    learning_rate=0.01
)

# Train model
trainer.train(
    X_train=X_train, 
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    batch_size=64,
    epochs=500,
    samples_nbr=3  # Number of Monte Carlo samples for ELBO
)
```

### Making Predictions with Uncertainty

```python
# Get predictions with uncertainty estimates
y_pred_mean, y_pred_std, all_predictions = trainer.predict(
    X_test, 
    num_samples=100  # More samples = better uncertainty estimates
)

# Confidence intervals (e.g., 95%)
ci_upper = y_pred_mean + (2 * y_pred_std)  # 2 std devs = ~95% CI
ci_lower = y_pred_mean - (2 * y_pred_std)
```

### Data Processing and Evaluation

```python
from Bayes.utils import prepare_data, evaluate_uncertainty

# Prepare data
X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_data(
    X, y, test_size=0.2, random_state=42, scale=True
)

# Evaluate predictions with uncertainty
metrics = evaluate_uncertainty(
    y_true=y_test,
    y_pred_mean=y_pred_mean,
    y_pred_std=y_pred_std,
    std_multiplier=2  # 2 standard deviations (~95% confidence interval)
)

print(f"Mean Squared Error: {metrics['mse']:.4f}")
print(f"R² Score: {metrics['r2']:.4f}")
print(f"Confidence Interval Accuracy (95%): {metrics['ci_accuracy']*100:.2f}%")
```

### Visualization

```python
from Bayes.utils import plot_predictions

plot_predictions(
    y_true=y_test,
    y_pred_mean=y_pred_mean,
    y_pred_std=y_pred_std,
    std_multiplier=2,  # 2 std devs = ~95% CI
    max_samples=100    # Limit for clearer visualization
)
```

## Advanced Usage

### Customizing Prior Distributions

You can customize the prior distribution for the Bayesian layers:

```python
from Bayes.modules import BayesianLinear

layer = BayesianLinear(
    in_features=10, 
    out_features=5,
    prior_sigma1=0.1,    # Standard deviation for the first Gaussian
    prior_sigma2=0.4,    # Standard deviation for the second Gaussian
    prior_pi=0.5         # Mixture weight
)
```

### Freezing the Network for Evaluation

```python
# Freeze network to use posterior means (deterministic prediction)
model.freeze()

# Make predictions using only the mean of the posterior
determinstic_preds = model(X_test)

# Unfreeze for sampling-based prediction with uncertainty
model.unfreeze()
```

## Theory: Bayes by Backprop

Bayes by Backprop (Blundell et al., 2015) is a variational inference method for training Bayesian Neural Networks. The key ideas are:

1. **Weight Uncertainty**: Instead of single weight values, the network learns a distribution over weights (usually Gaussian)
2. **Variational Inference**: Approximate the true posterior distribution with a simpler one (variational distribution)
3. **ELBO Loss**: Optimize the Evidence Lower Bound, which balances data fit against complexity:
   
   ```
   ELBO = E[log p(D|w)] - KL[q(w|θ) || p(w)]
   ```
   
   Where:
   - E[log p(D|w)] is the expected log-likelihood (data fit)
   - KL[q(w|θ) || p(w)] is the KL divergence between posterior and prior (complexity penalty)

4. **Monte Carlo Sampling**: Use random samples from the weight distributions during both training and inference

## References

- Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424). In Proceedings of the 32nd International Conference on Machine Learning.

## License

MIT
