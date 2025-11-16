import abc
from turtle import update
import torch
import torch.nn.functional as F
from catsample import sample_categorical

from model import utils as mutils

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass



@register_predictor(name="matu")
class MatuPredictor(Predictor):
    def init_timesteps(self, steps, l_noise_eps=1e-3, early_stopping_time=1e-5, device=torch.device('cpu')):
        self.bigT = -torch.log1p(-(1 - l_noise_eps) * 1) # since in LogLinearNoise, t: [1,0] -> total_noise: [-log(eps), 0] from noise_lib.py
        self.delta = early_stopping_time # early stopping time delta in MATU
        ret_timesteps = torch.linspace(0, self.bigT - self.delta, steps + 1, device=device)
        return ret_timesteps

    def update_fn(self, score_fn, x, t_prev, t_curr):
        K = self.graph.dim  # vocabulary size
        numK = (x==K-1).sum(dim=-1) # number of mask tokens
        beta = K * numK * 1.0 / (torch.exp(self.bigT - t_curr) - 1)
        innerN = torch.poisson(torch.tensor(beta*(t_curr - t_prev)))
        z = x.detach().clone()
        if innerN > 0: 
            tau_list = torch.rand(innerN) * (t_curr - t_prev) + t_prev
            for tau in tau_list:
                score = score_fn(z, self.bigT - tau) # socre at reverse time tau = score at forward time T - tau, SEDD paramterized with forward time.
                tilde_R = self.graph.reverse_rate(z, score) # [B, L, K] tilde_R[sample, i, j] = Prob(z[z_i->j]||z)
                tilde_Rz = tilde_R.sum(dim=(1, 2), keepdim=True)
                # calculate hat_R
                keep_prob = (tilde_Rz < beta).expand(-1, tilde_R.shape[1], tilde_R.shape[2]) # [B, L, K]
                factor_expanded = tilde_Rz.view(-1, 1, 1)
                hat_R = torch.where(keep_prob, tilde_Rz, tilde_Rz / factor_expanded * beta)
                # show which samples are kept
                sample_indicator = torch.zeros_like(tilde_Rz, dtype=torch.bool)
                update_above = tilde_Rz >= beta
                sample_indicator[update_above] = True
                update_else = ~update_above
                p_update = 1 - (tilde_Rz[update_above] / beta)
                rand_update_probs = torch.rand_like(p_update)
                sample_indicator[update_else] = rand_update_probs < p_update # [B,1] to show whether the sample is updated
                # update the sample
                b_indices = sample_indicator[:, 0].nonzero(as_tuple=True)[0]
                if b_indices.numel() == 0:
                    continue
                sub_hat_R = hat_R[b_indices]
                update_idx, seq_len, vocab_size = sub_hat_R.shape
                update_prob = sub_hat_R.view(update_idx, -1)
                update_prob = update_prob / update_prob.sum(dim=1, keepdim=True)
                update_pos_token_pairs = torch.multinomial(update_prob, 1).squeeze(dim=1)
                update_pos = update_pos_token_pairs // vocab_size
                update_token = update_pos_token_pairs % vocab_size
                z[b_indices, update_pos] = update_token
        return z

@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score) #  Eq. (7): Mid-term \eta * Q_t^\tok * score  = \eta * dsigma * Q^\to * score(x_t, sigma)
        x = self.graph.sample_rate(x, rev_rate) # Eq. (7): \delta function + reverse rate
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        return sample_categorical(probs)
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
    
    return sampling_fn
    

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device) # initial state [B = batch_size, L = model_max_length]
        
        if predictor == "matu":
            timesteps = predictor.init_timesteps(steps, eps, device=device)
        else:
            timesteps = torch.linspace(1, eps, steps + 1, device=device) # timesteps from 1 to eps \approx 0, corresponds to training time (intput time of DM) from -ln eps to -\ln(1-eps) \approx 0
        
        dt = (1 - eps) / steps

        for i in range(steps):
            x = projector(x)
            if predictor != "matu":
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=device) # [B , 1]
                x = predictor.update_fn(sampling_score_fn, x, t, dt)
            elif predictor == "matu" and i>0:
                x = predictor.update_fn(sampling_score_fn, x, timesteps[i], timesteps[i-1])
            

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
            
        return x
    
    return pc_sampler

