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
    def __init__(self, graph, noise, bigW=100000):
        super().__init__(graph, noise)
        self.bigT = None
        self.delta = None
        self.bigW = bigW
        self.bigK = graph.dim # this K = vocab size matches theoretical results in MATU
        self.actual_steps = 0

    def init_timesteps(self, steps, batch_dims, l_noise_eps=1e-3, early_stopping_time=1e-6, device=torch.device('cpu')):
        self.bigT = -torch.log1p(torch.tensor(l_noise_eps-1.0)) # since in LogLinearNoise, t: [1,0] -> total_noise: [-log(eps), 0] from noise_lib.py
        self.delta = early_stopping_time # early stopping time delta in MATU
        self.bigK = int(steps * 1.0 / batch_dims[1]) # related to rejection rate to control final complexity
        ret_timesteps = torch.linspace(0, self.bigT - self.delta, self.bigW + 1, device=device)
        return ret_timesteps

    def update_fn(self, score_fn, x, t_prev, t_curr, device=torch.device('cpu')):
        K = self.bigK  # hyper paras
        numK = (x==self.graph.dim-1).sum(dim=-1) # number of mask tokens
        beta = K * numK * 1.0 / (torch.exp(self.bigT - t_curr) - 1)
        innerN = int(torch.poisson(torch.tensor(beta*(t_curr - t_prev))).item())
        z = x.detach().clone()
        if innerN > 0: 
            self.actual_steps += innerN
            tau_list, _ = torch.sort(torch.rand(innerN).to(device) * (t_curr - t_prev) + t_prev)
            # print(f"t_curr: {t_curr}, actual steps: {self.actual_steps}")
            for tau in tau_list:
                score = score_fn(z, self.bigT - tau) # socre at reverse time tau = score at forward time T - tau, SEDD paramterized with forward time.
                tilde_R = self.graph.reverse_rate(z, score) # [B, L, K] tilde_R[sample, i, j] = Prob(z[z_i->j]||z)
                # calculate \tilde{R}(z) note to remove the Prob(z||z)
                tilde_R = tilde_R.clamp_min(0)
                tilde_Rz = tilde_R.sum(dim=(1, 2), keepdim=True).squeeze(-1) # [B, 1]
                # calculate hat_R
                B = tilde_R.shape[0]
                keep_prob = (tilde_Rz < beta) \
                    .reshape(B, 1, 1) \
                    .expand(B, tilde_R.shape[1], tilde_R.shape[2]) # [B, L, K]
                factor_expanded = tilde_Rz.view(-1, 1, 1)
                hat_R = torch.where(keep_prob, tilde_R, tilde_R / factor_expanded * beta)
                # show which samples are kept
                sample_indicator = torch.zeros_like(tilde_Rz, dtype=torch.bool)
                update_above = tilde_Rz >= beta
                sample_indicator[update_above] = True
                update_else = ~update_above
                p_update = (tilde_Rz[update_else] / beta)
                rand_update_probs = torch.rand_like(p_update)
                sample_indicator[update_else] = rand_update_probs < p_update # [B,1] to show whether the sample is updated
                # update the sample
                b_indices = sample_indicator[:, 0].nonzero(as_tuple=True)[0]
                if b_indices.numel() == 0:
                    continue
                sub_hat_R = hat_R[b_indices]    # [B_selected, L, K]
                position_prob = sub_hat_R.sum(dim=-1)
                position_sample = torch.multinomial(position_prob, 1) # [B_selected, 1]

                row_indices = torch.arange(sub_hat_R.shape[0], device=device) # [B_selected]
                position_idxs = position_sample.squeeze(dim=1)
                row_probs = sub_hat_R[row_indices, position_idxs, :]
                token_sample = torch.multinomial(row_probs, num_samples=1)

                position_sample = position_sample.squeeze(dim=1)
                token_sample = token_sample.squeeze(dim=1)
                z[b_indices, position_sample] = token_sample
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
        
        if type(predictor) is MatuPredictor:
            timesteps = predictor.init_timesteps(steps, batch_dims, device=device)
        else:
            timesteps = torch.linspace(1, eps, steps + 1, device=device) # timesteps from 1 to eps \approx 0, corresponds to training time (intput time of DM) from -ln eps to -\ln(1-eps) \approx 0
        
        dt = (1 - eps) / steps

        
        
        if type(predictor) is MatuPredictor:
            for i in range(predictor.bigW):
                x = predictor.update_fn(sampling_score_fn, x, timesteps[i], timesteps[i+1], device=device)
            print(f"actual steps: {predictor.actual_steps}")
        else:
            for i in range(steps):
                x = projector(x)         
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=device) # [B , 1]
                x = predictor.update_fn(sampling_score_fn, x, t, dt)
            if denoise:
                # denoising step
                x = projector(x)
                t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
                x = denoiser.update_fn(sampling_score_fn, x, t)  
            
        return x
    
    return pc_sampler

