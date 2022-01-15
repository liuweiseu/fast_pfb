import cuda_pfb
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,np.pi/2,0.1)
din = np.sin(x)

plt.plot(din)
