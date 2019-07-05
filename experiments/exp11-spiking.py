import track_deep_heat

for seed in range(20):
  for merge in [5]:
    for n_epochs in [200, 500]:
     for n_features_1 in [3, 10]: 
      track_deep_heat.TrackingTrial().run(dataset_dir='../../../davis_data',
        merge=merge,
        normalize=False,
        split_spatial=False,
        n_features_1=n_features_1, n_features_2=10,
        n_epochs=n_epochs, seed=seed,
        data_dir='exp11', data_format='npz',
        input_data='events',
        )
