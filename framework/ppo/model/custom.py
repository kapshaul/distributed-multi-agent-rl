import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


# Creating a masking matrix
def masking_matrix(m, n, p):
    """
    Creates an m x n binary matrix with exactly p fraction of zeros in each row.

    Args:
        m (int): Number of rows.
        n (int): Number of columns.
        p (float): Fraction of ones in each row (e.g., 0.3 means 30% zeros).

    Returns:
        torch.Tensor: An m x n matrix of 0s and 1s.
    """

    # Compute the number of zeros in each row.
    # We use round() to get as close as possible to the desired fraction.
    num_zeros = int(round((1-p) * m))
    # Make sure we do not have more zeros than the row length.
    num_zeros = min(num_zeros, m)

    # Create an m x n tensor initialized with zeros.
    matrix = torch.zeros(n, m, dtype=torch.int)
    for i in range(n):
        # Get a random permutation of column indices
        perm = torch.randperm(m)
        # Select indices that will become ones (the remaining ones after the zeros)
        one_indices = perm[num_zeros:]
        # Set those indices in the i-th row to 1.
        matrix[i, one_indices] = 1
    return matrix


# Feature Transformer
class FeatureTransform(nn.Module):
    def __init__(self, hidden_size=0, adjacency_matrix=None):
        super().__init__()

        if adjacency_matrix is not None:
            # GNN matrix
            F = torch.FloatTensor(self.gnn_normalize(adjacency_matrix))
            self.GNN = True
        else:
            # Scaler vector
            scale = torch.FloatTensor([np.sqrt(np.pi) / 2])
            F = torch.randn(hidden_size) * scale
            self.GNN = False

        # Register constant vector or matrix into the buffer
        self.register_buffer("F", F)

    def gnn_normalize(self, adjacency_matrix):
        # Add self-loops (optional, common in GNNs)
        adjacency_matrix = adjacency_matrix + np.eye(adjacency_matrix.shape[0])
        # Compute the degree matrix
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        # Compute D^(-1/2)
        degree_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))
        # Compute the normalized adjacency matrix
        normalized_adj = degree_inv_sqrt @ adjacency_matrix @ degree_inv_sqrt
        return normalized_adj

    def forward(self, x):
        if self.GNN:
            return torch.matmul(x, self.F)
        else:
            return x * self.F


# Customized linear weight matrix to mask
class CustomLinear(nn.Module):
    """
    A linear layer that applies a mask to its weight matrix before
    performing the linear transformation.
    """

    def __init__(self, in_features, out_features, p=1, m=0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Create the usual weight and (optionally) bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

        # Mask
        if p != 1.0:
            mask = masking_matrix(in_features, out_features, p)
            # Register the mask into the buffer
            self.register_buffer('mask', mask)
        else:
            self.mask = None

        # Scaler vector
        scale = torch.FloatTensor([np.sqrt(np.pi) / 2])
        #F = torch.randn(out_features) * scale
        #scale = torch.FloatTensor([2])
        F = torch.rand(out_features)# * scale
        F[torch.rand(out_features) < m] = 1.0

        # Register constant vector or matrix into the buffer
        self.register_buffer("F", F.unsqueeze(1))

    def reset_parameters(self):
        # Initialization weights, Kaiming uniform is a common choice
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        if self.bias is not None:
            # PyTorch's recommended uniform initialization for bias
            fan_in = self.in_features
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.mask is not None:
            # Element-wise multiply the weight by the mask
            weight = self.weight * self.mask
        else:
            weight = self.weight
        weight = weight * self.F
        return F.linear(x, weight, self.bias)


# Customized convolution weight matrix to mask
class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, m=0, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding

        # Initialize filters (weights) and biases
        self.weights = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        torch.nn.init.kaiming_normal_(self.weights, mode='fan_in', nonlinearity='relu')

        # Initialize scaling factor
        scale = torch.randn(out_channels, 1, 1, 1) * torch.FloatTensor([np.sqrt(np.pi) / 2])
        scale[torch.rand(out_channels) < m] = 1.0
        self.register_buffer("scale", scale)

    def forward(self, x):
        # Element-wise multiply the weight with scale factor
        scaled_w = self.weights * self.scale
        return F.conv2d(x, scaled_w, bias=self.bias, stride=self.stride, padding=self.padding)


# Customized multi-head attention
class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True):
        super(CustomMultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        # Projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        if not self.batch_first:
            # Convert (seq_len, batch, embed_dim) to (batch, seq_len, embed_dim)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, T, _ = query.size()

        # Linear projections
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape to (B, num_heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # Concatenate heads
        output = context.transpose(1, 2).contiguous().view(B, T, self.embed_dim)

        # Final projection
        #output = self.out_proj(context)

        if not self.batch_first:
            # Convert back to (seq_len, batch, embed_dim)
            output = output.transpose(0, 1)

        return output, attn_weights
