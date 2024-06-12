import torch
from torch import nn

class PatchEmbedding(nn.Module):
  '''
  Create patch embeddings using hybrid architecture from each image
  '''
  def __init__(
    self,
    in_channels: int,
    image_size: tuple,
    patch_size: int,
    embedding_dim: int
  ):
    super(PatchEmbedding, self).__init__()
    self.in_channels = in_channels
    self.image_height, self.image_width = image_size
    self.patch_size = patch_size
    # HW / (P^2)
    self.n_patches = (self.image_height * self.image_width) / (self.patch_size ** 2)
    # check that the image divides evenly into patches
    assert(self.n_patches.is_integer())
    self.embedding_dim = embedding_dim

    # sequence of patch embeddings
    self.embedding = nn.Sequential(
        nn.Conv2d(in_channels=self.in_channels, out_channels=self.embedding_dim, kernel_size=self.patch_size, stride=self.patch_size),
        nn.Flatten(start_dim=2, end_dim=3)
    )

    # class token used for final classification
    self.class_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
    # learned position embedding to encode relative ordering of patch embeddings
    self.position_embedding = nn.Parameter(torch.randn(1, int(self.n_patches + 1), self.embedding_dim))

  def forward(self, x):
    # [B, N, D]
    x = self.embedding(x).permute(0, 2, 1) 
    # prepend class token to patch embedding
    n_batches = x.shape[0]
    x = torch.concatenate([self.class_token.repeat(n_batches, 1, 1), x], dim=1)
    # add position embedding to patch embedding
    x += self.position_embedding
    return x


class MultiHeadSelfAttention(nn.Module):
  '''
  MSA layer as part of encoder block
  '''
  def __init__(
    self,
    embedding_dim: int,
    n_heads: int,
    dropout: float
  ):
    super(MultiHeadSelfAttention, self).__init__()
    self.embedding_dim = embedding_dim
    self.n_heads = n_heads
    self.dropout = dropout
    self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
    self.MSA = nn.MultiheadAttention(
      embed_dim=self.embedding_dim,
      num_heads=self.n_heads,
      dropout=self.dropout,
      batch_first=True
    )

  def forward(self, x): 
    x = self.layer_norm(x)
    x, attn = self.MSA(query=x, key=x, value=x)
    return x


class MLP(nn.Module):
  '''
  MLP layer as part of encoder block
  '''
  def __init__(
    self,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float
  ):
    super(MLP, self).__init__()
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.dropout = dropout
    self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
    self.MLP = nn.Sequential(
        nn.Linear(in_features=self.embedding_dim, out_features=self.hidden_dim),
        nn.GELU(),
        nn.Dropout(p=self.dropout),
        nn.Linear(in_features=self.hidden_dim, out_features=self.embedding_dim),
        nn.Dropout(p=self.dropout)
    )

  def forward(self, x):
    x = self.layer_norm(x)
    x = self.MLP(x)
    return x


class EncoderLayer(nn.Module):
  def __init__(
    self,
    embedding_dim: int,
    n_heads: int,
    attn_dropout: float,
    hidden_dim: int,
    dropout: float
  ):
    super(EncoderLayer, self).__init__()
    self.embedding_dim = embedding_dim
    self.n_heads = n_heads
    self.attn_dropout = attn_dropout
    self.hidden_dim = hidden_dim
    self.dropout = dropout

    self.MSA = MultiHeadSelfAttention(
      embedding_dim=self.embedding_dim,
      n_heads=self.n_heads,
      dropout=self.attn_dropout
    )

    self.MLP = MLP(
      embedding_dim=self.embedding_dim,
      hidden_dim=self.hidden_dim,
      dropout=self.dropout
    )

  def forward(self, x):
    x = self.MSA(x) + x
    x = self.MLP(x) + x
    return x
  

class ViT(nn.Module):
  def __init__(
    self,
    in_channels: int,
    num_classes: int,
    image_size: int,
    patch_size: int,
    embedding_dim: int,
    n_layers: int,
    n_heads: int,
    attn_dropout: float,
    hidden_dim: int,
    dropout: float
  ):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = num_classes
    self.image_height, self.image_width = image_size
    self.patch_size = patch_size
    self.embedding_dim = embedding_dim
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.attn_dropout = attn_dropout
    self.hidden_dim = hidden_dim
    self.dropout = dropout
    
    # image patch embeddings
    self.patch_embedding = PatchEmbedding(
      in_channels=self.in_channels,
      image_size=(self.image_height, self.image_width),
      patch_size=self.patch_size,
      embedding_dim=self.embedding_dim
    )

    # encoder
    encoder_layers = [EncoderLayer(
        embedding_dim=self.embedding_dim,
        n_heads=self.n_heads,
        attn_dropout=self.attn_dropout,
        hidden_dim=self.hidden_dim,
        dropout=self.dropout
      ) for i in range(self.n_layers)]
    self.encoder = nn.Sequential(*encoder_layers)
      
    # classification head
    self.mlp_head = nn.Sequential(
      nn.LayerNorm(normalized_shape=self.embedding_dim),
      nn.Linear(in_features=self.embedding_dim, out_features=self.out_channels),
    )

  def forward(self, x):
    x = self.patch_embedding(x)
    x = self.encoder(x)
    # classification done using class token embedding of each image
    x = self.mlp_head(x[:, 0, :])

    return x






