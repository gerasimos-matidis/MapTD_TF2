def fit(model, train_dataset, test_dataset, summary_writer,
        total_steps=2**20, step_to_reduce_lr=2**17, enumerate_from=0):
    initial_lr = 1e-4
    decay_rate = 0.1
    opt = tf.keras.optimizers.Adam(
        learning_rate=StepDecayLRSchedule(initial_lr, decay_rate, 
                                          step_to_reduce_lr), 
        epsilon=1e-8)
    
    training_start = time.time()
    start = training_start
    for step, (tile, score_map, geo_map, training_mask) in \
                train_dataset.repeat().take(total_steps).\
                enumerate(start=enumerate_from):
        
        training_loss = train_step(model, opt, tile, 
                                    score_map, geo_map, training_mask,
                                    step, summary_writer)
        
        step = step.numpy()
        
        if step % 500 == 0:
            display.clear_output(wait=True)
            
            print(f'Step {step}/{total_steps}')
            
            if step != 0:
                print(f'Time taken for the last 500 steps: '
                      f'{time.time()-start:.2f} sec')
                estimated_remaining_time = int((time.time() - training_start) / \
                    step * (total_steps - step))
                print(f'Estimated time for the training to finish: '
                      f'{estimated_remaining_time // 3600} hrs, '
                      f'{int(estimated_remaining_time % 3600 / 60)} mins')
                
            print('Current Learning Rate: ', opt.lr(step).numpy())
            print(f'Training loss: {training_loss:.4f}')
            example_tile, example_score_map, _, _ = next(iter(test_ds.take(1)))
            example_pred_score_map, _ = maptd(example_tile, training=True)
            show_score_map(example_tile, 
                           example_score_map, 
                           example_pred_score_map, 
                           threshold=0.7)             
            start = time.time()
               
        if step == step_to_reduce_lr:
            new_lr = opt.lr(step).numpy()
            print(f'The learning rate for the optimizer was decreased from '
                  f'{initial_lr} to {new_lr}')
            
        if (step + 1) % 10000 == 0:
            ckpt.save(file_prefix=ckpt_prefix)