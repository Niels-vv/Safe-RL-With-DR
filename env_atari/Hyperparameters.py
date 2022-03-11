config = {
            # Learning
            'train_q_per_step' : 4,
            'gamma' : 0.99,
            'train_q_batch_size' : 256,
            'batches_before_training' : 200,
            'target_q_update_frequency' : 10000,
            'lr' : 0.00025,
            'plot_every' : 10,
            'decay_steps' : 500000,
            'max_gradient_norm' : 10,
            'intermediate_results_freq' : 100       # Store intermediate results in drive every x episodes
        }