import numpy as np
import torch
from pathlib import Path
from spectpsftoolbox.kernel1d import ArbitraryKernel1D, FunctionKernel1D
from spectpsftoolbox.operator2d import GaussianOperator, Rotate1DConvOperator, RotateSeperable2DConvOperator
import pytomography
device = pytomography.device

psf_operator_dict = {
    'GI-HEGP @ 440 keV': 'params_440keV_GIHEGP_256shift_sparse_DW_pyrex66_1DR',
    'SY-HE @ 440 keV': 'params_440keV_SYHE_256shift_sparse_DW_pyrex66_1DR'
}

def getScriptPath():
    try:
        script_path = Path(__file__).resolve().parents[1]
    except:
        from inspect import getsourcefile
        script_path = Path(getsourcefile(lambda: 0)).resolve().parents[1]
    finally:
        return script_path


def get_psf_operator(psf_name):
    script_path = getScriptPath()
    path = script_path / 'Resources' / 'Data' / 'PSF_models' / f'{psf_operator_dict[psf_name]}_'
    path = str(path)
    gauss_amplitude_params = torch.load(path+'gauss_amplitude.pt')
    gauss_sigma_params = torch.load(path+'gauss_sigma.pt')
    tail_amplitude_params = torch.load(path+'tail_amplitude.pt')
    tail_sigma_params = torch.load(path+'tail_sigma.pt')
    bkg_amplitude_params = torch.load(path+'bkg_amplitude.pt')
    bkg_sigma_params = torch.load(path+'bkg_sigma.pt')
    Nx0 = 255
    tail_kernel_decay = 0.5
    if "_SY" in path:
        rot = 90
        dx0 = 0.24
    elif "_GI" in path:
        rot = 0
        dx0 = 0.2208336
    a_min = 1.
    a_max = 55.
    # Gaussian component
    gauss_amplitude_fn = lambda a, bs: bs[0]*torch.exp(-a*bs[1]) + bs[2]*torch.exp(-a*bs[3])
    gauss_sigma_fn = lambda a, bs: bs[0] + bs[1]*(torch.sqrt(a**2 + bs[2]**2) - torch.abs(bs[2]))
    gaussian_operator = GaussianOperator(
        gauss_amplitude_fn,
        gauss_sigma_fn,
        gauss_amplitude_params,
        gauss_sigma_params,
    )
    # Tail component
    Nx_tail = round(np.sqrt(2)*Nx0)
    if Nx_tail%2==0:
        Nx_tail += 1
    x = torch.arange(-(Nx_tail-1)/2, (Nx_tail+1)/2, 1).to(device) * dx0
    tail_kernel = torch.tensor(torch.exp(-tail_kernel_decay*torch.abs(x)), requires_grad=True, device=device)
    tail_amplitude_fn = lambda a, bs: bs[0]*torch.exp(-a*bs[1]) + bs[2]*torch.exp(-a*bs[3])
    tail_sigma_fn = lambda a, bs, a_min=a_min: 1 + bs[0]*(torch.sqrt((a-a_min)**2 + bs[1]**2) - torch.abs(bs[1]))
    tail_kernel1D = ArbitraryKernel1D(tail_kernel, tail_amplitude_fn, tail_sigma_fn, tail_amplitude_params, tail_sigma_params, dx0, grid_sample_mode='bicubic')
    tail_operator = Rotate1DConvOperator(
        tail_kernel1D,
        N_angles = 3,
        additive=True,
        rot=rot,
    )
    # Bkg component
    bkg_amplitude_fn = lambda a, bs: bs[0]*torch.exp(-a*bs[1]) + bs[2]*torch.exp(-a*bs[3])
    bkg_sigma_fn = lambda a, bs: bs[0] + bs[1]*(torch.sqrt(a**2 + bs[2]**2) - torch.abs(bs[2]))
    kernel_fn = lambda x: torch.exp(-torch.abs(x))
    bkg_kernel1D = FunctionKernel1D(kernel_fn, bkg_amplitude_fn, bkg_sigma_fn, bkg_amplitude_params, bkg_sigma_params, a_min=a_min, a_max=a_max)
    bkg_operator = RotateSeperable2DConvOperator(
        bkg_kernel1D,
        N_angles = 1,
        additive=False
    )
    # Operator
    psf_operator = (tail_operator + bkg_operator) * gaussian_operator + gaussian_operator
    return psf_operator