# TensorTonic Solutions

My solutions to [TensorTonic](https://tensortonic.com) problems: core machine
learning and deep learning algorithms implemented **from scratch** with NumPy,
without ML libraries.

Solutions are synced automatically from the platform as I solve them.

## Solutions

| Problem | Topic | Approach | Solution |
|---|---|---|---|
| Sigmoid (NumPy) | Activation functions | Vectorized, numerically stable sigmoid: uses `exp(x) / (1 + exp(x))` for negative inputs to avoid overflow | [sigmoid-numpy.py](sigmoid-numpy/sigmoid-numpy.py) |
| Logistic Regression Training | Classification | Batch gradient descent on binary cross-entropy: gradients `Xᵀ(p − y)/N` for weights and `mean(p − y)` for bias | [logistic-regression-training.py](logistic-regression-training/logistic-regression-training.py) |

## Why from scratch?

Writing the maths by hand (gradients, numerical stability, vectorization)
builds the intuition that libraries hide. Every solution here uses only NumPy.

## Running a solution

```bash
pip install numpy
python -c "
import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('s', 'sigmoid-numpy/sigmoid-numpy.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(m.sigmoid([-1000, 0, 1000]))
"
```

## License

[MIT](LICENSE). Problem statements belong to TensorTonic; the solution code is mine.
