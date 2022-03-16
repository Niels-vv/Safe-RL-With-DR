config = {
            # Learning
            'train_q_per_step' : 1,
            'gamma' : 0.97,
            'train_q_batch_size' : 32,
            'batches_before_training' : 313,
            'target_q_update_frequency' : 1000,
            'lr' : 5e-5,
            'multi_step' : 1,
            'plot_every' : 1,
            'eps_end' : 0.02,
            'decay_steps' : 400000,
            'max_gradient_norm' : 10,
            'intermediate_results_freq' : 20       # Store intermediate results in drive every x episodes
        }