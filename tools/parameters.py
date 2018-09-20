param = {'data': {
  'normalise': True,
  'window_size': 50,
  'train_test_split': 0.2},
 'model': {'layers': [{'input_timesteps': 49,
    'neurons': 100,
    'return_seq': True,
    'type': 'lstm'},
   {'rate': 0.2, 'type': 'dropout'},
   {'neurons': 100, 'return_seq': True, 'type': 'lstm'},
   {'neurons': 100, 'return_seq': False, 'type': 'lstm'},
   {'rate': 0.2, 'type': 'dropout'},
   {'activation': 'linear', 'neurons': 1, 'type': 'dense'}],
  'loss': 'mse',
  'optimizer': 'adam'},
 'training': {'batch_size': 32, 'epochs': 1000}}