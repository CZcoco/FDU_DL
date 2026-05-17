from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        scale = np.sqrt(2.0 / in_dim)
        self.W = initialize_method(size=(in_dim, out_dim)) * scale
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.params['W'] + self.params['b']

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.params['W'].T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer using im2col for efficient computation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size)) * scale
        self.b = np.zeros((out_channels,))

        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.col = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        """
        self.input = X
        N, C_in, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1

        if p > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        else:
            X_padded = X

        col = np.zeros((N, C_in * k * k, H_out * W_out))
        for i in range(H_out):
            for j in range(W_out):
                patch = X_padded[:, :, i*s:i*s+k, j*s:j*s+k]
                col[:, :, i * W_out + j] = patch.reshape(N, -1)

        self.col = col

        W_reshaped = self.params['W'].reshape(self.out_channels, -1)
        output = np.matmul(W_reshaped, col)
        output = output.reshape(N, self.out_channels, H_out, W_out)
        output += self.params['b'].reshape(1, -1, 1, 1)

        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        N, C_out, H_out, W_out = grads.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding
        _, C_in, H, W = self.input.shape

        grad_reshaped = grads.reshape(N, C_out, -1)

        self.grads['b'] = np.sum(grads, axis=(0, 2, 3))

        dW = np.matmul(grad_reshaped, self.col.transpose(0, 2, 1))
        self.grads['W'] = np.sum(dW, axis=0).reshape(self.params['W'].shape)

        W_reshaped = self.params['W'].reshape(C_out, -1)
        dcol = np.matmul(W_reshaped.T, grad_reshaped)

        if p > 0:
            dX_padded = np.zeros((N, C_in, H + 2 * p, W + 2 * p))
        else:
            dX_padded = np.zeros((N, C_in, H, W))

        for i in range(H_out):
            for j in range(W_out):
                patch_grad = dcol[:, :, i * W_out + j].reshape(N, C_in, k, k)
                dX_padded[:, :, i*s:i*s+k, j*s:j*s+k] += patch_grad

        if p > 0:
            dX = dX_padded[:, :, p:-p, p:-p]
        else:
            dX = dX_padded

        return dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        N, C, H, W = X.shape
        k, s = self.kernel_size, self.stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1
        output = np.zeros((N, C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                region = X[:, :, i*s:i*s+k, j*s:j*s+k]
                output[:, :, i, j] = region.max(axis=(2, 3))
        return output

    def backward(self, grads):
        N, C, H, W = self.input.shape
        k, s = self.kernel_size, self.stride
        H_out, W_out = grads.shape[2], grads.shape[3]
        dX = np.zeros_like(self.input)
        for i in range(H_out):
            for j in range(W_out):
                region = self.input[:, :, i*s:i*s+k, j*s:j*s+k]
                max_val = region.max(axis=(2, 3), keepdims=True)
                mask = (region == max_val)
                dX[:, :, i*s:i*s+k, j*s:j*s+k] += mask * grads[:, :, i:i+1, j:j+1]
        return dX


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        batch_size = predicts.shape[0]
        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts
        log_probs = np.log(self.probs + 1e-12)
        loss = -np.sum(log_probs[np.arange(batch_size), labels]) / batch_size
        one_hot = np.zeros_like(self.probs)
        one_hot[np.arange(batch_size), labels] = 1.0
        self.grads = (self.probs - one_hot) / batch_size
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition