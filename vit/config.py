data_params = {
  'in_channels' : 3,
  'num_classes' : 10,
  'image_size' : (32, 32),
  'patch_size' : 4,
  'embedding_dim' : 256
}

model_params = {
  'n_layers' : 3,
  'n_heads' : 4,
  'attn_dropout' : 0.0,
  'hidden_dim' : 256,
  'dropout' : 0.1
}

train_params = {
  'epochs' : 50,
  'batch_size' : 32,
  'lr' : 0.001,
  'momentum' : 0.9
}