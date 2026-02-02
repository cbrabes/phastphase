import torch
from torch.fft import fftn, ifftn
def deconvolution(far_field, reference):
    reference_autocorr = ifftn(torch.square(torch.abs(fftn(reference, s = far_field.shape))))
    total_correlation = ifftn(far_field)
    corrected_correlation = total_correlation-reference_autocorr
    corrected_correlation = ifftn(fftn(corrected_correlation)/fftn(reference,s=far_field.shape))