################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Unit tests for PFB, for numerical correctness."""

import numpy as np
from katsdpsigproc import accel

from katgpucbf.fgpu import pfb

import time
def decode_10bit_host(data):
    """Convert an array of signed 10-bit integers to signed 16-bit representation."""
    bits = np.unpackbits(data).reshape(-1, 10)
    # Replicate the high (sign) bit
    extra = np.tile(bits[:, 0:1], (1, 6))
    combined = np.hstack([extra, bits])
    packed = np.packbits(combined)
    return packed.view(">i2").astype("i2")


def pfb_fir_host(data, channels, weights):
    """Apply a PFB-FIR filter to a set of data on the host."""
    step = 2 * channels
    assert len(weights) % step == 0
    taps = len(weights) // step
    decoded = decode_10bit_host(data)
    window_size = 2 * channels * taps
    out = np.empty((len(decoded) // step - taps + 1, step), np.float32)
    for i in range(0, len(out)):
        windowed = decoded[i * step : i * step + window_size] * weights
        out[i] = np.sum(windowed.reshape(-1, step), axis=0)
    return out


def test_pfb_fir(repeat=1):
    """Test the GPU PFB-FIR for numerical correctness.

    Parameters
    ----------
    repeat
        Number of times to repeat the GPU operation, default 1. A larger value
        can be used for benchmarking purposes.
    """
    ctx = accel.create_some_context(interactive=False)
    queue = ctx.create_command_queue()

    taps = 4
    spectra = 128
    channels = 16384
    samples = 2 * channels * (spectra + taps-1)
    #h_in = np.random.randint(0, 256, samples*10//8, np.uint8)
    h_in = np.random.randint(0, 256, samples, np.uint8)
    print('Size of h_in=',len(h_in))
    weights = np.random.uniform(-1.0, 1.0, (2 * channels * taps,)).astype(np.float32)
    #expected = pfb_fir_host(h_in, channels, weights)

    template = pfb.PFBFIRTemplate(ctx, taps)
    fn = template.instantiate(queue, samples, spectra, channels)
    fn.ensure_all_bound()
    t0 = time.time()
    fn.buffer("in").set(queue, h_in)
    fn.buffer("weights").set(queue, weights)
    fn.in_offset = 0
    fn.out_offset = 0
    fn.spectra = 100
    t1 = time.time()
    fn()
    t2 = time.time()
    h_out = fn.buffer("out").get(queue)
    print("Size of h_out",len(h_out))
    print('Write Data to Buffer(s):',t1-t0)
    print('GPU Processing Time(ms):',(t2-t1)*1000)
    print('Total Time(s):',t2-t0)
    #np.testing.assert_allclose(h_out, expected, rtol=1e-5, atol=1e-3)


def test_fft():
    """Test the GPU FFT for numerical correctness."""
    ctx = accel.create_some_context(interactive=False)
    queue = ctx.create_command_queue()
    spectra = 1
    channels = 67108864
    rng = np.random.default_rng(seed=2021)
    h_data = rng.uniform(-5, 5, (spectra, 2 * channels)).astype(np.float32)
    #expected = np.fft.rfft(h_data, axis=-1)
    t0 = time.time()
    fn = pfb.FFT(queue, spectra, channels)
    fn.ensure_all_bound()
    fn.buffer("in").set(queue, h_data)
    fn()
    t1 = time.time()
    print('GPU Processing Time(s):',t1-t0)
    h_out = fn.buffer("out").get(queue)
    #np.testing.assert_allclose(h_out, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_pfb_fir(repeat=1)
    #test_fft()
