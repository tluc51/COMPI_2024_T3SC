from t3sc.data.noise_models import *

def radial_profile_corrected(data):
    # Center of the data
    center = np.array([(i//2) for i in data.shape])
    y, x = np.indices((data.shape))  # First get the indices of the pixels
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)  # Compute the radial distance for each pixel
    r = r.astype(np.int64)  # Corrected data type

    # Averaging over concentric circles
    tbin = np.bincount(r.ravel(), data.ravel())  # Sum of values in each bin
    nr = np.bincount(r.ravel())  # Number of elements in each bin
    radialprofile = tbin / nr
    return radialprofile

def noise_to_radial_power_spectrum(data):
    data_fft = torch.fft.fft2(data)
    data_fft_shifted = torch.fft.fftshift(data_fft)
    power_spectrum = torch.abs(data_fft_shifted)**2
    
    return radial_profile_corrected(power_spectrum.detach().numpy())