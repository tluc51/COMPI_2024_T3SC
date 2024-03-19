import logging

import numpy as np
import torch

from t3sc.data.splits import icvl_rgb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseNoise:
    def __init__(self, test=None, ssl=None, seed=None):
        self.name = type(self).__name__
        self.test = test
        self.ssl = ssl
        self.seed = seed

    def update_sigmas(self):
        raise NotImplementedError

    def apply(self, x):
        raise NotImplementedError

    def params_str(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        msg = f"{self.name}_{self.params_str()}_b{self.bands}_s{self.seed}"
        return msg


class ConstantNoise:
    def __init__(self, sigma, bands, **kwargs):
        self.bands = bands
        self.sigma = sigma

        self.sigmas = self.sigma * torch.ones((self.bands, 1, 1)) / 255
        self.sigma_avg = self.sigma / 255

    def apply(self, x, seed, **kwargs):
        generator = np.random.RandomState(seed=seed)
        noise_pixels = torch.tensor(
            generator.randn(*x.shape),
            dtype=torch.float32,
            device=x.device,
        )

        noisy = x + self.sigmas * noise_pixels

        return x, noisy.float()

    def __repr__(self) -> str:
        name = type(self).__name__
        msg = f"{name}_s{self.sigma}"
        return msg


class UniformNoise:
    def __init__(self, sigma_min, sigma_max, **kwargs):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def apply(self, x, seed):
        generator = np.random.RandomState(seed=seed)
        sigmas = torch.tensor(
            generator.rand(x.shape[0], 1, 1),
            dtype=torch.float32,
            device=x.device,
        )
        noise_pixels = torch.tensor(
            generator.randn(*x.shape),
            dtype=torch.float32,
            device=x.device,
        )

        sigmas = self.sigma_min + sigmas * (self.sigma_max - self.sigma_min)
        self.sigmas = sigmas / 255
        self.sigma_avg = (self.sigma_min + self.sigma_max) / (2 * 255)
        noisy = x + self.sigmas * noise_pixels

        return x, noisy

    def __repr__(self) -> str:
        name = type(self).__name__
        msg = f"{name}_min{self.sigma_min}_max{self.sigma_max}"
        return msg


class CorrelatedNoise(BaseNoise):
    def __init__(self, beta, eta, bands, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.eta = eta
        self.bands = bands
        self.update_sigmas()

    def update_sigmas(self):
        idx = np.arange(self.bands)

        sigmas = self.beta * np.exp(
            -((idx / self.bands - 0.5) ** 2) / (4 * self.eta ** 2)
        )
        self.sigmas = torch.tensor(sigmas / 255, dtype=torch.float32).view(
            -1, 1, 1
        )

        self.sigma_avg = self.sigmas.mean()

    def apply(self, x, **kwargs):
        noisy = x + self.sigmas * torch.randn(*x.shape)
        return x, noisy

    def params_str(self):
        return f"beta{self.beta}_eta{self.eta}"


class StripesNoise:
    def __init__(
        self,
        bands,
        ratio_bands=0.33,
        ratio_columns=[0.05, 0.15],
        stripe_intensity=0.5,
        sigma=25,
        **kwargs,
    ):
        self.bands = bands
        self.ratio_bands = ratio_bands
        self.ratio_columns = ratio_columns
        self.stripe_intensity = stripe_intensity
        self.sigma = sigma
        self.std = self.sigma / 255
        self.sigmas = torch.ones(self.bands) * self.std
        self.sigma_avg = self.std

    def stripe_noise(self, c, h, w):
        stripe_noise = torch.zeros((c, 1, w))

        n_bands = int(c * self.ratio_bands)
        bands_affected = self.generator.permutation(c)[:n_bands]
        n_cols = self.generator.randint(
            int(w * self.ratio_columns[0]),
            int(w * self.ratio_columns[1]),
            (len(bands_affected),),
        )
        logger.debug(f"Bands affected : {bands_affected}")
        logger.debug(f"N stripes col : {n_cols}")
        for i, band_idx in enumerate(bands_affected):
            col_idx = self.generator.permutation(w)[: n_cols[i]]
            stripe_noise[band_idx, :, col_idx] = (
                torch.rand(len(col_idx)) - 0.5
            ) * self.stripe_intensity

        return stripe_noise.float()

    def gaussian_noise(self, c, h, w):
        pixel_noise = self.std * torch.tensor(self.generator.randn(c, h, w))
        return pixel_noise.float()

    def apply(self, x, seed, **kwargs):
        self.generator = np.random.RandomState(seed=seed)
        s_noise = self.stripe_noise(*x.shape)
        g_noise = self.gaussian_noise(*x.shape)
        noisy = x + g_noise + s_noise
        return x, noisy

    def __repr__(self):
        name = type(self).__name__
        return (
            f"{name}_rb{self.ratio_bands}-rc{self.ratio_columns[0]}-"
            f"{self.ratio_columns[1]}_"
            f"s{self.sigma}"
        )


class ColorNoise:
    def __init__(
        self,
        alpha,   # S(f) ~ 1/f^alpha
        sigma=25,
        **kwargs,
    ):
        self.alpha = alpha
        self.sigma = sigma
        self.std = self.sigma/255
        
    def generate_noise(self, shape):
        c, h, w = shape
        assert len(shape) == 3, "Shape must be a 3-tuple corresponding to (channels, height, width)."
        
        # Create a grid of frequencies
        freqs = torch.fft.fftfreq(max(h, w), d=1.0).to(torch.float32)
        freqs = torch.sqrt(freqs[None, :]**2 + freqs[:, None]**2)
        
        # Apply the 1/f^alpha filter in the frequency domain
        with torch.no_grad():
            filter = 1.0 / (freqs**self.alpha + 1e-8)  # Avoid division by zero
            filter /= filter.max()  # Normalize
        
        # Generate white noise and apply the filter
        white_noise = torch.randn((c, h, w), dtype=torch.float32) * self.std
        color_noise = torch.fft.ifft2(torch.fft.fft2(white_noise) * filter).real
        
        return color_noise.float()
    
    def apply(self, x, seed=None, **kwargs):
        if seed is not None:
            torch.manual_seed(seed)
        
        p_noise = self.generate_noise(x.shape)
        noisy = x + p_noise
        
        return x, noisy

    def __repr__(self):
        return f"ColorNoise_alpha{self.alpha}_sigma{self.sigma}"
    
    
class PinkNoise(ColorNoise):
    def __init__(self, alpha=1, sigma=25, **kwargs):
        super().__init__(alpha, sigma, **kwargs)


class BrownianNoise(ColorNoise):
    def __init__(self, alpha=2, sigma=25, **kwargs):
        super().__init__(alpha, sigma, **kwargs)
        
        
class BlueNoise(ColorNoise):
    def __init__(self, alpha=-1, sigma=25, **kwargs):
        super().__init__(alpha, sigma, **kwargs)

# Occlusion with Gaussian noise
class Occlusion:
    def __init__(self, nb_bands_affected, occlusion_size, sigma, bands, **kwargs):
        self.bands = bands
        self.nb_bands_affected = nb_bands_affected
        self.occlusion_size = occlusion_size

        self.sigma = sigma

        self.sigmas = self.sigma * torch.ones((self.bands, 1, 1)) / 255
        self.sigma_avg = self.sigma / 255

    def apply(self, x, seed, **kwargs):
        generator = np.random.RandomState(seed=seed)

        c, h, w = x.shape

        band_idx = np.array(range(c))

        # For ICVL only to visualize
        # RGB: [8, 14, 27] always occluded bands

        band_idx = np.delete(band_idx, icvl_rgb)

        bands_affected = generator.choice(band_idx, self.nb_bands_affected-3, replace=False)
        bands_affected = np.append(bands_affected, icvl_rgb)

        max_y_pos = w - self.occlusion_size[1]
        max_x_pos = h - self.occlusion_size[0]

        x_pos = generator.choice(range(max_x_pos))
        y_pos = generator.choice(range(max_y_pos))

        mask = torch.ones(*x.shape)
        mask[bands_affected, x_pos:x_pos+self.occlusion_size[0], y_pos:y_pos+self.occlusion_size[1]] = 0

        occluded = mask*x

        # Add noise to occluded image
        noise_pixels = torch.tensor(
            generator.randn(*x.shape),
            dtype=torch.float32,
            device=x.device,
        )

        occluded = occluded + self.sigmas * noise_pixels

        return x, occluded

    def __repr__(self) -> str:
        name = type(self).__name__
        msg = f"{name}_nb-band-affected{self.nb_bands_affected}_occlusion-size{self.occlusion_size}_s{self.sigma}"
        return msg