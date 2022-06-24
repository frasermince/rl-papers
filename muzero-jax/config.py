from jaxline import base_config
from ml_collections import config_dict


def get_config(debug: bool = False) -> config_dict.ConfigDict:
  """Get Jaxline experiment config."""
  config = base_config.get_base_config()
  config.random_seed = 42
  # E.g. '/data/pretrained_models/k0_seed100' (and set k_fold_split_id=0, below)
  config.restore_path = config_dict.placeholder(str)
  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              debug=debug,
              optimizer=dict(
                  optimizer='adam',
                  weight_decay=1e-5,
                  adam_kwargs=dict(b1=0.9, b2=0.999),
                  learning_rate_schedule=dict(
                      use_schedule=True,
                      base_learning_rate=0.05,
                      warmup_steps=50000,
                      total_steps=config.get_ref('training_steps'),
                  ),
              ),
		        training=dict(
              batch_size=128

              ),
        )))

  ## Training loop config.
#   config.training_steps = 500000
  config.checkpoint_dir = '/tmp/checkpoint/muzero-jax/'
  config.train_checkpoint_all_hosts = False
#   config.log_train_data_interval = 10
#   config.log_tensors_interval = 10
#   config.save_checkpoint_interval = 30
#   config.best_model_eval_metric = 'accuracy'
#   config.best_model_eval_metric_higher_is_better = True

  return config