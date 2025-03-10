# Bayes by Backprop

A Bayesian Neural Network framework for regression tasks implemented in PyTorch. This framework provides tools for uncertainty estimation in neural networks through variational inference.

## Installation

```bash
pip install bayes-regression
```

Then install the required dependencies:

```bash
pip install git+https://github.com/zchccsx/Bayes-by-Backprop.git
```

## Usage

### Creating a Model

```python
import Bayes
from Bayes import BayesianLinear, BayesianRegressor

# Create a Bayesian neural network
model = BayesianRegressor(input_dim=10, hidden_dims=[32, 16], output_dim=1)

### Training the Model

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
