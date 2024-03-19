from utils import *
import matplotlib.pyplot as plt

c, h, w = 3, 100, 100
x = torch.zeros((c, h, w))

# Define instances
pink_generator = PinkNoise()
brownian_generator = BrownianNoise()
blue_generator = BlueNoise()


# Generate noise
_, pink_noisy = pink_generator.apply(x, seed=123)
_, brownian_noisy = brownian_generator.apply(x, seed=123)
_, blue_noisy = blue_generator.apply(x, seed=123)


# Convert noise back to freq domain
radial_power_spectrum_pink = noise_to_radial_power_spectrum(pink_noisy[0,:,:])
radial_power_spectrum_brownian = noise_to_radial_power_spectrum(brownian_noisy[0,:,:])
radial_power_spectrum_blue = noise_to_radial_power_spectrum(blue_noisy[0,:,:])


plt.figure()
frequencies = np.arange(len(radial_power_spectrum_pink))
plt.figure(figsize=(10, 6))
plt.loglog(frequencies, radial_power_spectrum_pink, color="pink", label="Pink Noise")
plt.loglog(frequencies, radial_power_spectrum_brownian, color="brown", label="Brownian Noise")
plt.loglog(frequencies, radial_power_spectrum_blue, color="blue", label="Blue Noise")
plt.title('Radial Power Spectrum')
plt.xlabel('Radial Frequency')
plt.ylabel('Power')
plt.legend(loc="best")

# Show 2D-noise
plt.figure(figsize=(8,8))
plt.subplot(131)
plt.imshow(pink_noisy[0,:,:])
plt.title("Pink Noise")

plt.subplot(132)
plt.imshow(brownian_noisy[0,:,:])
plt.title("Brownian Noise")

plt.subplot(133)
plt.imshow(blue_noisy[0,:,:])
plt.title("Blue Noise")

plt.show()