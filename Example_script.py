#### A script to show how to use each of the 4 functions in BTS.
import BTS
import numpy as np


#### Example to show how to fit a single spectrum

# Read in the relevant parameter file
param = BTS.read_parameters("./Fit_single_line.param")

# Construct a noisy spectrum for the test
velocity = np.linspace(-2,2,100)

amplitude = 1.0
centroid = 0.2
sigma = 0.4
noise_level = 0.1

np.random.seed(1)
spectrum = amplitude*np.exp(-(velocity-centroid)**2/(2*sigma**2))
spectrum = spectrum + np.random.normal(loc=0.0,scale=noise_level,size=100)
mask = np.zeros_like(spectrum)
mask[spectrum>0.4] = 1

# Fit the noisy spectrum and report the co-efficients and their errors
co_eff, errors, AIC = BTS.fit_single_line(velocity,spectrum,mask,param)
print("The fitted amplitude = %.3lf p/m %.3lf, and the actual ampltiude = %.3lf" %(co_eff[0],errors[0],amplitude))
print("The fitted centroid  = %.3lf p/m %.3lf, and the actual centroid  = %.3lf" %(co_eff[1],errors[1],centroid))
print("The fitted width     = %.3lf p/m %.3lf, and the actual width     = %.3lf" %(co_eff[2],errors[2],sigma))


#### Example to show how to make moment maps from a fits cube and then fit the entire cube

# Read in the relevant parameter file
param = BTS.read_parameters("./Fit_cube.param")

# Run the function to make the moments using the moment-masking technique
BTS.make_moments(param)

# Using the generated mask, fit the entire datacube
BTS.fit_a_fits(param)


#### Example to show how to run the single Gaussian test, checking the typical errors on the fitting parameters

# Read in the relevant parameter file
param = BTS.read_parameters("./Single_gaussian_test.param")

# Run the test which reports the errors on the amplitude, centroid and width
BTS.single_gaussian_test(param)


#### Example to show how to run the multiple Gaussian test, checking how accurate the number of velocity components is

# Read in the relevant parameter file
param = BTS.read_parameters("./Multi_gaussian_test.param")

# Run the test which reports the number of spectrum fit with the incorrect number of components
BTS.multi_gaussian_test(param)
