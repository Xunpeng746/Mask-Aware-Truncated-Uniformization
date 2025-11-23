import abc
from turtle import update
import torch
import torch.nn.functional as F
from transformers.models.qwen3_next.modeling_qwen3_next import torch_recurrent_gated_delta_rule
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

@register_predictor(name="matu_1")
class MatuPredictor1(Predictor):
    def __init__(self, graph, noise, bigW_factor=3.0, early_stopping_time=1e-5, bigT=1.0):
        super().__init__(graph, noise)
        self.bigT = 1.0
        self.delta = early_stopping_time
        self.bigW = int(bigW_factor * self.bigT / self.delta)
        self.bigK = graph.dim # this K = vocab size matches theoretical results in MATU
        self.actual_steps = 0
        self.num_ideal_beta = 0

    def init_timesteps(self, steps, batch_dims, device=torch.device('cpu')):
        self.bigK = int(steps * 1.0 / batch_dims[1])  # related to rejection rate to control final complexity
        return torch.linspace(0, self.bigT - self.delta, self.bigW + 1, device=device)

    def update_fn(self, score_fn, x, t_prev, t_curr, device=torch.device('cpu')):
        K = self.bigK  # hyper paras
        numK = 1.0 * (x==self.graph.dim-1).sum(dim=-1) # number of mask tokens
        sigma, dsigma = self.noise(self.bigT - t_curr) # sigma(1-t) and partial sigma(1-t)/partial t
        beta = dsigma * K * numK * 1.0 / (torch.exp(sigma) - 1)
        innerN = int(torch.poisson(torch.tensor(beta*(t_curr - t_prev))).item())
        # print(f"t_prev: {t_prev}, t_curr: {t_curr}, beta: {beta}, innerN: {innerN}")
        avg_transition_prob = 0.0
        if innerN > 0: 
            self.actual_steps += innerN
            tau_list, _ = torch.sort(torch.rand(innerN).to(device) * (t_curr - t_prev) + t_prev)
            for tau in tau_list:
                inner_sigma, inner_dsigma = self.noise(self.bigT - tau)
                inner_score = score_fn(x, inner_sigma).to(dtype=torch.float32)
                tilde_R = inner_dsigma * self.graph.reverse_rate(x, inner_score)
                tilde_R = tilde_R.clamp_min(0)
                tilde_Rz = tilde_R.sum(dim=(1, 2), keepdim=True).squeeze(-1) # [1, 1]
                keep_prob = (tilde_Rz < beta) \
                    .reshape(1, 1, 1) \
                    .expand(1, tilde_R.shape[1], tilde_R.shape[2]) # [1, L, K]
                transition_prob = tilde_Rz.item() /  beta.item()
                
                # for debug
                avg_transition_prob += transition_prob
                hat_R = torch.where(keep_prob, tilde_R, tilde_R / transition_prob)
                rand_update_probs = torch.rand(1).item()
                if transition_prob > 1.0 or rand_update_probs < transition_prob:
                    '''
                    self.num_ideal_beta += 1
                    position_prob = hat_R.sum(dim=-1)
                    position_sample = torch.multinomial(position_prob, 1).squeeze(dim=1) # [1]
                    row_probs = hat_R[0, position_sample, :]
                    token_sample = torch.multinomial(row_probs, num_samples=1).squeeze(dim=1) # [1]
                    x[0, position_sample] = token_sample
                    self.num_ideal_beta += 1 
                    '''
                    # Rewrite Categorical Sampling Gumble-max trick
                    gumbel_norm = 1e-10 - (torch.rand_like(hat_R) + 1e-10).log()
                    max_indices = (hat_R / gumbel_norm).argmax() % (x.shape[1] * self.graph.dim)
                    position_sample = max_indices // self.graph.dim
                    token_sample = max_indices % self.graph.dim
                    x[0, position_sample] = token_sample
                    
                    # print(f"t: {1-t_prev.item()}, position_denoised: {position_sample.item()}, value_denoised: {token_sample.item()}, prob_values: {tilde_R[0][position_sample.item()][token_sample.item()].item()}, max_prob_value: {torch.max(tilde_R).item()}")

        return x

                


@register_predictor(name="matu")
class MatuPredictor(Predictor):
    def __init__(self, graph, noise, bigW=100000):
        super().__init__(graph, noise)
        self.bigT = None
        self.delta = None
        self.bigW = bigW
        self.bigK = graph.dim # this K = vocab size matches theoretical results in MATU
        self.actual_steps = 0
        self.num_ideal_beta = 0

    def init_timesteps(self, steps, batch_dims, l_noise_eps=1e-3, early_stopping_time=1e-3, device=torch.device('cpu')):
        self.bigT = -torch.log1p(torch.tensor(l_noise_eps-1.0)) # since in LogLinearNoise, t: [1,0] -> total_noise: [-log(eps), 0] from noise_lib.py
        self.delta = early_stopping_time # early stopping time delta in MATU
        self.bigK = int(steps * 1.0 / batch_dims[1]) # related to rejection rate to control final complexity
        ret_timesteps = torch.linspace(0, self.bigT - self.delta, self.bigW + 1, device=device)
        return ret_timesteps

    def update_fn(self, score_fn, x, t_prev, t_curr, device=torch.device('cpu')):
        K = self.bigK  # hyper paras
        # K = 2 * self.bigT / torch.log(self.bigT + 1.0) * 1.0 / (self.bigT - t_curr+1.0)
        numK = (x==self.graph.dim-1).sum(dim=-1) # number of mask tokens
        beta = K * numK * 1.0 / (torch.exp(self.bigT - t_curr) - 1)
        innerN = int(torch.poisson(torch.tensor(beta*(t_curr - t_prev))).item())
        print(f"t_prev: {t_prev}, t_curr: {t_curr}, beta: {beta}, innerN: {innerN}")
        if innerN > 0: 
            self.actual_steps += innerN
            tau_list, _ = torch.sort(torch.rand(innerN).to(device) * (t_curr - t_prev) + t_prev)
            # print(f"t_curr: {t_curr}, actual steps: {self.actual_steps}")
            for tau in tau_list:
                score = score_fn(x, self.bigT - tau).to(dtype=torch.float32) # socre at reverse time tau = score at forward time T - tau, SEDD paramterized with forward time.
                tilde_R = self.graph.reverse_rate(x, score) # [B, L, K] tilde_R[sample, i, j] = Prob(z[z_i->j]||z)
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

                if tilde_Rz.item() <= beta.item():
                    self.num_ideal_beta += 1
                
                sub_hat_R = hat_R[b_indices]    # [B_selected, L, K]
                position_prob = sub_hat_R.sum(dim=-1)
                position_sample = torch.multinomial(position_prob, 1) # [B_selected, 1]

                row_indices = torch.arange(sub_hat_R.shape[0], device=device) # [B_selected]
                position_idxs = position_sample.squeeze(dim=1)

                # before:
                row_probs = sub_hat_R[row_indices, position_idxs, :]
                token_sample = torch.multinomial(row_probs, num_samples=1)

                # row_probs = sub_hat_R[row_indices, position_idxs, :]  # [B_selected, K]
                
                # # If using absorbing graph, exclude mask token (last token) from sampling
                # # to ensure generated tokens are within valid vocab range
                # if self.graph.absorb:
                #     # Exclude mask token (graph.dim - 1) from sampling
                #     vocab_size = self.graph.dim - 1  # actual vocab size without mask token
                #     row_probs_vocab = row_probs[..., :vocab_size]  # [B_selected, vocab_size]
                #     # Renormalize probabilities
                #     row_probs_vocab = row_probs_vocab / row_probs_vocab.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                #     token_sample = torch.multinomial(row_probs_vocab, num_samples=1)
                # else:
                #     token_sample = torch.multinomial(row_probs, num_samples=1)

                position_sample = position_sample.squeeze(dim=1)
                token_sample = token_sample.squeeze(dim=1)
                x[b_indices, position_sample] = token_sample
                
        return x

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
    

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, hyper_params, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    if predictor == "matu_1":
        predictor = MatuPredictor1(graph, noise, bigW_factor=hyper_params['bigW'], early_stopping_time=hyper_params['delta'], bigT=1.0)
    else:
        predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims) # initial state [B = batch_size, L = model_max_length]
        x = x.to(device)
        
        if type(predictor) is MatuPredictor1:
            timesteps = predictor.init_timesteps(steps, batch_dims, device=device)
        elif type(predictor) is MatuPredictor:
            timesteps = predictor.init_timesteps(steps, batch_dims, device=device)
        else:
            timesteps = torch.linspace(1, eps, steps + 1, device=device) # timesteps from 1 to eps \approx 0, corresponds to training time (intput time of DM) from -ln eps to -\ln(1-eps) \approx 0
        
        dt = (1 - eps) / steps

        
        if type(predictor) is MatuPredictor1:
            for i in range(predictor.bigW):
                x = predictor.update_fn(sampling_score_fn, x, timesteps[i], timesteps[i+1], device=device)
            t_prev = timesteps[predictor.bigW-1]
            step_size = timesteps[predictor.bigW-1] - timesteps[predictor.bigW-2]
            while ((x == graph.dim-1).sum().item() > 0) and 1-t_prev > 5e-7:
                t_curr = min(t_prev + step_size, (predictor.bigT+t_prev)/2.0)
                # print(f"t_prev: {t_prev}, t_curr: {t_curr}, Mask Num: {(x == graph.dim-1).sum().item()}")
                x = predictor.update_fn(sampling_score_fn, x, t_prev, t_curr, device=device)
                t_prev = t_curr
            print(f"Actual Steps: {predictor.actual_steps}")
            print("---------")
        elif type(predictor) is MatuPredictor:
            for i in range(predictor.bigW):
                x = predictor.update_fn(sampling_score_fn, x, timesteps[i], timesteps[i+1], device=device)
            t_prev = timesteps[predictor.bigW-1]
            step_size = timesteps[predictor.bigW-1] - timesteps[predictor.bigW-2]
            while ((x == graph.dim-1).sum().item() > 0):
                t_curr = min(t_prev + step_size, (predictor.bigT+t_prev)/2.0)
                # print(f"t_prev: {t_prev}, t_curr: {t_curr}, Mask Num: {(x == graph.dim-1).sum().item()}")
                x = predictor.update_fn(sampling_score_fn, x, t_prev, t_curr, device=device)
                t_prev = t_curr
            print(f"Num of ideal beta: {predictor.num_ideal_beta}")
            predictor.num_ideal_beta = 0
            print("---------")
        else:
            for i in range(steps):
                x = projector(x)         
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=device) # [B , 1]

                x = predictor.update_fn(sampling_score_fn, x, t, dt)
            
            print(f"Mask Num: {(x == graph.dim-1).sum().item()}")
            if denoise:
                # denoising step
                x = projector(x)
                t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
                x = denoiser.update_fn(sampling_score_fn, x, t)  
            
        return x
    
    return pc_sampler

