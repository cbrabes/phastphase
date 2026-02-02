import torch
from torch.jit import Final
from typing import Tuple, List, Optional
from torch.fft import ifft2, fft2
from phastphase import SOS_loss2
import scipy
import math
class WindingNumEvaluator(torch.nn.Module):
    image_size:         Final[Tuple[int, int]]
    support:            Final[Tuple[int, int]]
    cepstrum_size:      Final[Tuple[int, int]]
    oversample_ratio:   Final[int]
    def __init__(self, y, support: Tuple[int, int], oversample_ratio: Optional[int]=10, mask: Optional[torch.Tensor] = None):
        super().__init__()
        torch_device = y.device
        self.y = y.detach().clone()
        self.support = (support[0], support[1])
        self.image_size = (y.shape[0], y.shape[1])
        self.cepstrum_size = (y.shape[0]*oversample_ratio, y.shape[1]*oversample_ratio)
        self.oversample_ratio = oversample_ratio
        if mask is not None:
            self.support_mask = mask.detach().clone()
        else:
            self.support_mask = torch.ones(self.support, dtype=torch.cdouble, device=torch_device)
        self.cepstrum = torch.zeros(self.cepstrum_size, dtype=torch.cdouble, device=torch_device)
        self.x = torch.zeros(self.support, dtype=torch.cdouble, device=torch_device)
        self.autocorr = torch.zeros_like(self.cepstrum)
        
        self.autocorr[0:self.image_size[0], 0:self.image_size[1]] =torch.fft.ifftshift( ifft2(y))
        self.autocorr = torch.roll(self.autocorr, (-self.image_size[0]//2, -self.image_size[1]//2), dims=(0,1))

        self.cepstrum = ifft2(torch.log(fft2(self.autocorr)/(oversample_ratio**2) + 1e-12))
        self.mask = torch.zeros_like(self.cepstrum)
        self.x_out = torch.zeros_like(self.cepstrum)

    def forward(self, winding: Tuple[int, int]):
        sz = self.cepstrum.shape
        
        (n,m) = torch.meshgrid(torch.fft.ifftshift(torch.linspace(-math.ceil(sz[0]/2), sz[0]//2 ,steps=sz[0])),
                            torch.fft.ifftshift(torch.linspace(-math.ceil(sz[1]/2), sz[1]//2 ,steps=sz[1])), indexing='ij')
        self.mask = torch.zeros_like(self.cepstrum)
        if winding == (0,0):
            self.mask[0:sz[0]//2, 0:sz[1]//2] = 2
            self.mask[0,0] =1
        elif winding[0] == 0:
            self.mask[0:sz[0]//2, 0:sz[1]] = 2
            self.mask[0,0] =1
        elif winding[1] == 0:
            self.mask[0:sz[0], 0:sz[1]//2] = 2
            self.mask[0,0] =1
        else:
            self.mask[winding[1]*n + winding[0]*m >= 0] = 2
            self.mask = torch.roll(self.mask, (winding[0], winding[1]), dims=(0, 1))
            self.mask[0:2*winding[0]+1, 0:2*winding[1]+1]=1
            self.mask = torch.roll(self.mask, (-winding[0], -winding[1]), dims=(0, 1))
        self.x_out = ifft2(torch.exp(1/2*fft2(torch.mul(self.mask, self.cepstrum))),norm='ortho')
        self.x_out = torch.roll(self.x_out, (winding[0], winding[1]), dims=(0, 1))
        self.x = self.x_out[0:self.support[0], 0:self.support[1]]
        self.x  = self.x/torch.exp(1j*torch.angle(self.x[winding[0], winding[1]]))
        return torch.square(torch.linalg.vector_norm(torch.square(torch.abs(fft2(self.support_mask*self.x, s=self.image_size, norm='ortho'))) - self.y))
    
def create_wind_eval(y, support: Tuple[int, int], oversample_ratio: Optional[int]=10, mask: Optional[torch.Tensor] = None):
    return torch.jit.optimize_for_inference(torch.jit.script(WindingNumEvaluator(y, support, oversample_ratio, mask).eval()))

def basin_refine(support, y, winding, x0,niters=100):

    ytorch = torch.as_tensor(y)
    xshape = torch.zeros(support)
    def loss_func(x): return SOS_loss2(torch.view_as_complex(x.view_as(xshape)), torch.sqrt(ytorch),1,winding[0],winding[1])


    loss_grad = torch.func.grad(loss_func)

    def hvp(x, p):
        return torch.autograd.functional.vhp(loss_func, x, p)[1].t()


    def scipy_loss(x):
        torch_in = torch.from_numpy(x)
        return loss_func(torch_in).numpy(force=True)

    def scipy_grad(x):
        torch_in = torch.from_numpy(x)
        return loss_grad(torch_in).numpy(force=True)
    def scipy_hvp(x,p):
        torch_x = torch.from_numpy(x)
        torch_p = torch.from_numpy(p)
        return hvp(torch_x, torch_p).numpy(force=True)

    return scipy.optimize.basinhopping(scipy_loss, x0,stepsize=1,niter=niters,interval=10, T = 0,minimizer_kwargs={'method':'TNC', 'jac' : scipy_grad, 'hessp' : scipy_hvp, 'tol': 1e-7, 'options': {'maxiter': 100}},disp=True )
