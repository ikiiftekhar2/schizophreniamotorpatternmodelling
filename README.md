# Schizophrenia Motor Activity Prediction

Predicting motor activity in schizophrenia patients using time series and clinical data.

---

## Dataset

We use the publicly available dataset described in:

> Chavez-Badiola A, ... et al. **A multimodal dataset for deep phenotyping of psychosis (OBF dataset)**. *Scientific Data* (2025). [https://www.nature.com/articles/s41597-025-04384-3](https://www.nature.com/articles/s41597-025-04384-3)

The raw data includes:

* **schizophrenia-info.csv**: patient metadata (age, gender, clinical scores, medication).
* **schizophrenia folder**: per-patient CSVs with `timestamp` and `activity` readings.

---

## Problem Statement

We aim to predict future motor activity levels in schizophrenia patients given:

1. **Past activity timestamps**
2. **Previous activity readings** (`prev_act`)
3. **Categorical and numerical patient features**

Accurate predictions can help in early detection of motor abnormalities and personalized treatment.

---

## Data Preprocessing

1. **Metadata encoding**: Label-encoding of categorical fields (`gender`, `schtype`, `migraine`, etc.).
2. **Time-series assembly**: Merging each patient’s `timestamp` and `activity`, adding `prev_act` as a lag feature.
3. **Scaling**: Min–Max scaling of numerical features (`days`, `bprs`, `prev_act`) and target (`activity`).
4. **Splitting**: Per-patient 70% train, 15% validation, 15% test.

Final processed sets and scalers are saved as joblib files:

```
train_set.joblib
val_set.joblib
test_set.joblib
feature_scaler.joblib
activity_scaler.joblib
```

---

## Baseline Model: Naive Bayes

We first established a baseline using a Gaussian Naive Bayes regressor on the full feature set. On the test set:

```
Mean RMSE across patients: 106.7258
Mean MAE  across patients: 68.5046
```

---

## Recurrent Neural Network (RNN) Model

### Rationale

RNNs capture sequential patterns in time series by passing hidden states through time steps.

### Architecture

* **SimpleRNN** with 2 layers, hidden size 32, dropout 0.2
* **Dropout** on RNN outputs and FC layer
* **Packed sequences** to handle variable-length series

### Outcome

The RNN overfit quickly due to limited training samples (one sequence per patient) and complexity, leading to poor generalization.

---

## Multi-Layer Perceptron (MLP) Model

### Approach

A fixed-size sliding-window MLP:

* **Window size**: 10 past timesteps flattened into one vector
* **Architecture**: Dense layers (128 → 64 → 1) with ReLU and dropout
* **Loss**: MSE, **Optimizer**: Adam with L2 weight decay
* **Early stopping** based on validation loss

### Differences from RNN

* No recurrence: treats windows independently
* Fixed input dimension simplifies training
* Many training examples via sliding windows improves robustness

---

## Results (MLP)

```
Global Test RMSE: 0.0158
Global Test MAE:  0.0085
```

---

## Usage

Install dependencies and run scripts as needed. Below is an example of loading the trained MLP model and running predictions in a new Python file:

```python
import numpy as np
import pandas as pd
import torch
from joblib import load

# 1. Load data & scalers
X_test, y_test, t_test = load('test_set.joblib')
activity_scaler       = load('activity_scaler.joblib')

# 2. Prepare features
X_feat = X_test.drop(columns=['patient_id'])

# 3. Define MLP architecture (must match training)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# 4. Instantiate model and load weights
input_dim = X_feat.shape[1]
model = MLP(input_dim)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# 5. Predict scaled values
with torch.no_grad():
    X_tensor = torch.from_numpy(X_feat.values.astype(np.float32))
    y_pred_scaled = model(X_tensor).numpy().reshape(-1,1)

# 6. Inverse transform to original scale
y_pred = activity_scaler.inverse_transform(y_pred_scaled).flatten()

df_results = pd.DataFrame({
    'patient_id': X_test['patient_id'],
    'timestamp':  t_test,
    'y_true':     activity_scaler.inverse_transform(y_test.values.reshape(-1,1)).flatten(),
    'y_pred':     y_pred
})

# 7. Evaluate per-patient metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
metrics = []
for pid, grp in df_results.groupby('patient_id'):
    rmse = np.sqrt(mean_squared_error(grp['y_true'], grp['y_pred']))
    mae  = mean_absolute_error(grp['y_true'], grp['y_pred'])
    metrics.append({'patient_id': pid, 'rmse': rmse, 'mae': mae})

df_metrics = pd.DataFrame(metrics)
print(f"Mean RMSE across patients: {df_metrics['rmse'].mean():.4f}")
print(f"Mean MAE  across patients: {df_metrics['mae'].mean():.4f}")
```

---

## License

Data: Creative Commons Attribution 4.0 International via Nature

Code: MIT License
