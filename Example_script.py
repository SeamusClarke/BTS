#### A script to show how to use each of the 4 functions in BTS.

import BTS
import numpy as np

##### First run the single Gaussian test

#BTS.single_gaussian_test("./Example_files/SingleGaussianTest.param")

##### Next run the multiple Gaussian test

#BTS.multi_gaussian_test("./Example_files/MultipleGaussianTest.param")

##### Fit a single test spectrum

# Construct a noisy spectrum
velocity = np.linspace(-2,2,100)

amplitude = 1.0
centroid = 0.2
sigma = 0.4
noise_level = 0.1

np.random.seed(1)
spectrum = amplitude*np.exp(-(velocity-centroid)**2/(2*sigma**2))
spectrum = spectrum + np.random.normal(loc=0.0,scale=noise_level,size=100)

# Read the parameters
param = BTS.ReadParameters("./Example_files/SingleSpectrumFit.param")

# Fit the spectrum
co_eff,r_chi = BTS.fit_single_line(velocity,spectrum,param)

# Output results
print "The input amplitude, centroid and width were: %.2f, %.2f, %.2f" %(amplitude, centroid, sigma)
print "The output amplitude, centroid and width are: %.2f, %.2f, %.2f" %(co_eff[0], co_eff[1],co_eff[2])

##### Fit the test fits cube

# This should produce 4 fits files with the ampltiudes, centroids, widths and reduced chi_squareds.

#BTS.fit_a_fits("./Example_files/FitsFileFit.param")


