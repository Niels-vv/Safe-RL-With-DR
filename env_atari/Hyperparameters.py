config = {
            # Learning
            'train_q_per_step' : 4,
            'gamma' : 0.99,
            'train_q_batch_size' : 256,
            'batches_before_training' : 20,
            'target_q_update_frequency' : 250,
            'lr' : 0.0001,
            'plot_every' : 10,
            'decay_steps' : 100000,
            'intermediate_results_freq' : 100       # Store intermediate results in drive every x episodes
        }