import torch
import numpy as np

# mostly referenced huggingface sample code
# DDPM: Denoising Diffusion Probabilistic Models

class DDPMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.000085, beta_end: float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0) # [a0, a0*a1,...]
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inferenece_steps = num_inference_steps
        
        step_ratio = self.num_training_steps // self.num_inferenece_steps
        # 999, 999-step_ratio, 999-2*step_ratio,... 0
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::1]
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        # subtract step_ratio
        prev_t = timestep - (self.num_inferenece_steps // self.num_inferenece_steps)
        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)
        
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # compute variance according to formula (7)
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        # ensure variance > 0
        variance = torch.clamp(variance, min=1e-20)

        return variance
    
    def set_strength(self, strength=1):
        # skip initial steps from pure noise if strength < 1
        start_step = self.num_inferenece_steps - int(self.num_inferenece_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # compute predicted original sample according to formula (15)
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # compute the coefficients for pred_original_sample and current sample x_t with formula (7)
        pred_original_sample_coeff = alpha_prod_t_prev ** 0.5 * beta_prod_t / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        # compute the predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        variance = 0
        if t > 0:
            # only add noise if not last timestep
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise
        
        # N(0,1) -> N(mu, sigma^2)
        # X = mu + sigma * Z
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample

    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device)
        
        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # unsqueeze until enough dimensions
        while len(sqrt_alpha_prod.shape < len(original_samples.shape)):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # st_dev
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # unsqueeze until enough dimensions
        while len(sqrt_one_minus_alpha_prod.shape < len(original_samples.shape)):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # according to equation (4) of DDPM paper
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha_prod) * noise
        return noisy_samples