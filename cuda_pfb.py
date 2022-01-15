from asyncio import SubprocessTransport
import numpy as np
from katsdpsigproc import accel
from katgpucbf.fgpu import compute

def cuda_pfb(h_in):

    spectra_per_heap = 128
    taps = 4
    spectra = 512
    channels = 16384
    samples = 2 * channels * (spectra + taps-1)
    in_offset = 0
    out_offset = 0
    
    weights = np.ones((2 * channels * taps,)).astype(np.float32)
    try:
        h_in = np.frombuffer(h_in, dtype=np.uint8)
    except:
        pass

    ctx = accel.create_some_context()
    queue = ctx.create_command_queue()
    template = compute.ComputeTemplate(ctx, taps)
    fn = template.instantiate(queue, samples, spectra, spectra_per_heap, channels)
    fn.ensure_all_bound()
    
    # set data and weights to 'in' and 'weights' buffer
    fn.buffer('in0').set(queue,h_in)
    fn.pfb_fir[0].buffer('weights').set(queue,weights)
    #fn.buffer('in1').set(queue,h_in)
    #fn.pfb_fir[1].buffer('weights').set(queue,weights)

    fn.ensure_all_bound()
    fn.pfb_fir[0].in_offset = in_offset
    fn.pfb_fir[0].out_offset = out_offset
    fn.pfb_fir[0].spectra = spectra
    fn.pfb_fir[0]()
    '''
    fn.pfb_fir[0].in_offset = in_offset
    fn.pfb_fir[0].out_offset = out_offset
    fn.pfb_fir[0].spectra = spectra
    fn.pfb_fir[0]()
    '''
    fn.ensure_all_bound()
    fn.fft[0]()
    #fn.fft[1]()
    fft_out0 = fn.buffer('fft_out0').get(queue)
    #fft_out1 = fn.buffer('fft_out1').get(queue)
    fft_out0[:,0] = 0
    fft_out0_flat = fft_out0.flatten()
    return abs(fft_out0_flat).tolist()

def test(d):
    print("Hello!")
    d = np.frombuffer(d, dtype=np.uint8)
    print(type(d))
    print(d.shape)
    print(d)
    res = np.ndarray((10,)).astype(np.float32)
    print(res)
    return res.tolist()
    