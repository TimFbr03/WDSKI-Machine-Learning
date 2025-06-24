# Neuronal Network using Numpy  

To train the neuronal network, for the amount of Epochs, a forward pass, loss, and backward pass is done.

## Forward Pass
$$
    a^{[2]} = \sigma (W^{[2]} (\phi(W^{[1]} x + b^{[1]})) + b^{[2]})
$$
  
Schrittweise umstzung in Python
$$
    a^{[1]} = W^{[1]} x + b^{[1]}
$$
$$
    \phi(x) = \max(0, x)
$$
```python
    self.Z1 = np.dot(self.W1, X.T) + self.b1
    self.A1 = fn.relu(self.Z1)
```
  
<br>

$$
    a^{[2]} = \sigma ( W^{[2]} a^{[1]} + b^{[2]})  
$$
$$
    \sigma(z) = \frac{1}{1 + e^{-z}}
$$
```python
    self.Z2 = np.dot(self.W2, self.A1) + self.b2
    self.A2 = fn.sigmoid(self.Z2)
```

## Mean squared Error
$$
    MSE = \frac{1}{m} \sum^m_{i=1}(y_i - \hat{y}_i)^2
$$
```python
    mean_squared_error = np.mean((y_true - y_pred) ** 2)
```

## Backward Pass
```python
    m = X.shape[0]
    y = y.reshape(1, -1)
```
**m**: Anzahl der eingesetzten Datenpunkte  
**y**: y Transponieren  
  
### Error at output
$$
    \delta^{[2]} = \hat{y} - y  
$$
$$
    \frac{\partial \mathcal{L}}{\partial W^{[2]}} = \delta^{[2]} * (a^{[1]})^\top
$$
$$
    \frac{\partial \mathcal{L}}{\partial b^{[2]}} = \frac{1}{m} * \sum \delta^{[2]}
$$
```python
    dZ2 = self.A2 - y
    dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
```

### Error hidden Layer
$$
    \delta^{[1]} = ({w^{[2]}}^\top * \delta^{[2]}) \circ \phi(Z^{[1]}) 
$$

```python
    dA1 = np.dot(self.W2.T, dZ2)
    dZ1 = dA1 * fn.relu(self.Z1, True)
```

### Gradient hidden Layer
$$
    \frac{\partial \mathcal{L}}{\partial W^{[1]}} =\frac{1}{m} * (\delta^{[1]} * X)
$$
$$
    \frac{\partial \mathcal{L}}{\partial b^{[1]}} = \frac{1}{m} * \sum \delta^{[1]}
$$

```python
    dW1 = (1 / m) * np.dot(dZ1, X)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
```

### Gradient Descent
$$
    \theta^{t+1} \leftarrow \theta^t - \alpha \Delta f(\theta^t)
$$
with lr (learning rate) = $\alpha$
```
    self.W1 -= self.lr * dW1
    self.b1 -= self.lr * db1
    self.W2 -= self.lr * dW2
    self.b2 -= self.lr * db2
```