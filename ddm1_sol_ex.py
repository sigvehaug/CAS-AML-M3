# e.g. try different scheduler

from diffusers import DDIMScheduler

scheduler2 = DDIMScheduler.from_pretrained("google/ddpm-cat-256")
scheduler2.set_timesteps(10)  # number of diffusion steps


# then an update step will be:

with torch.no_grad():
    mod_out = model(x, t)
    noisy_residual = mod_out.sample  # model predicts noise step

    # scheduler step outputs a dictionary with 2 things:
    # 1-step sample update
    # and extrapolation to fully denoised sample
    x_scaled = scheduler2.scale_model_input(x, t)
    #print(x_scaled.shape, x.shape)
    ddpm_sched_out_dict = scheduler2.step(noisy_residual, t, x_scaled)
    previous_noisy_sample = ddpm_sched_out_dict.prev_sample
    pred_orig_sample = ddpm_sched_out_dict.pred_original_sample
    
    
 # do you notice a qualitative change in the samples?