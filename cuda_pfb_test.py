from asyncio import SubprocessTransport
import numpy as np
from katsdpsigproc import accel
from katgpucbf.fgpu import compute
import time
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
    
    t0 = time.time()
    
    ctx = accel.create_some_context()
    queue = ctx.create_command_queue()
    template = compute.ComputeTemplate(ctx, taps)
    fn = template.instantiate(queue, samples, spectra, spectra_per_heap, channels)
    fn.ensure_all_bound()
    
    t1 = time.time()

    # set data and weights to 'in' and 'weights' buffer
    fn.buffer('in0').set(queue,h_in)
    fn.pfb_fir[0].buffer('weights').set(queue,weights)
    fn.ensure_all_bound()

    t2 = time.time()

    fn.pfb_fir[0].in_offset = in_offset
    fn.pfb_fir[0].out_offset = out_offset
    fn.pfb_fir[0].spectra = spectra
    fn.pfb_fir[0]()

    t3 = time.time()

    fn.ensure_all_bound()
    fn.fft[0]()

    t4 = time.time()

    fft_out0 = fn.buffer('fft_out0').get(queue)

    t5 = time.time()

    print("%-35s: %f ms"%("   Time for creating context", (t1-t0)*1000))
    print("%-35s: %f ms"%("   Time for setting data", (t2-t1)*1000))
    print("%-35s: %f ms"%("   Time for pfb fir", (t3-t2)*1000))
    print("%-35s: %f ms"%("   Time for pfb fft", (t4-t3)*1000))
    print("%-35s: %f ms"%("   Time for copyting result out", (t5-t4)*1000))

    fft_out0[:,0] = 0
    fft_out0_flat = fft_out0.flatten()
    return abs(fft_out0_flat)