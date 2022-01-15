import numpy as np
from katsdpsigproc import accel
from katgpucbf.fgpu import compute
import time

N_POLS = 2
spectra_per_heap = 128
taps = 4
spectra = 512
channels = 16384
samples = 2 * channels * (spectra + taps-1)
in_offset = 0
out_offset = 0
ctx = accel.create_some_context()
queue = ctx.create_command_queue()
template = compute.ComputeTemplate(ctx, taps)
fn = template.instantiate(queue, samples, spectra, spectra_per_heap, channels)
fn.ensure_all_bound()
h_in = np.ones(samples).astype(np.uint8)
weights = np.ones((2 * channels * taps,)).astype(np.float32)
# set data and weights to 'in' and 'weights' buffer
t0 = time.time()
fn.buffer('in0').set(queue,h_in)
fn.pfb_fir[0].buffer('weights').set(queue,weights)
fn.buffer('in1').set(queue,h_in)
fn.pfb_fir[1].buffer('weights').set(queue,weights)

print('size of h_in(2 pols):%d'%(len(h_in)))

t1 = time.time()
fn.run_frontend([fn.buffer('in0'), fn.buffer('in1')],[in_offset, in_offset], out_offset,spectra)
#fn.run_backend()
fn.ensure_all_bound()
for fft_op in fn.fft:
    fft_op()
#fn.postproc()
#out = fn.postproc.buffer('out').get(queue)
t2 = time.time()

fft_out0 = fn.buffer('fft_out0').get(queue)
fft_out1 = fn.buffer('fft_out1').get(queue)
print("size of fft_out0",fft_out0.shape)
print("size of fft_out1",fft_out1.shape)
#print(fft_out0)
#print(fft_out1)

print('Write Data to Buffer(s):',t1-t0)
print('GPU Processing Time(ms):',(t2-t1)*1000)
print('Total Time(s):',t2-t0)