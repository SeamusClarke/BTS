![alt text](https://github.com/SeamusClarke/BTS/blob/master/Images/BTS.png)

# The BTS fitting code

The BTS (Behind The Spectrum) fitting code is a python module designed to be a fully-automated multiple-component fitter for optically-thin spectra. The code is open-source and can be downloaded here. If the code is used in published work please cite Clarke et al. 2018, MNRAS, 479, 1722, which shows the first use of the code, as well as a description of the code's methodology and tests of the code's accuracy.

The latest version (updated 18/07/2023) is officially BTS v2.0 as significant changes have been made to how model selection is done, as well new features being added such as moment-masking. The following documentation has been updated to reflect this change.

## Dependencies 

BTS requires 4 common libraries to be installed:

* Numpy,
* Astropy,
* Matplotlib,
* Scipy, at least version 0.17. 

To allow the import of the BTS module from any directory use the export command to modified the PYTHONPATH variable. This is done by adding the line

```
export PYTHONPATH=$PYTHONPATH:"Path to BTS' download location"
```
to the .bashrc file in the home directory.

## How to use the code

There are only 6 functions which the user will call:

* **read_parameters** - The function to read the parameter file and store the parameters in a dictionary. Its input is the parameter file name.
* **fit_single_line** - The function to fit a single spectrum. It takes 4 arguments: the velocity array, the intensity in these bins, a mask denoting which channels potentially have signal, and the parameters.
* **fit_a_fits** - The function to fit an entire fits file. It takes only 1 argument, the parameters. For this function to be used the fits file must have a header which contains information about the velocity range of the spectrum.
* **make_moments** - The function to make moment 0, 1 and 2 maps using the moment-masking technique. It takes only one input, the parameters.
* **single_gaussian_test** - A test function which produces test spectra containing a single Gaussian component. This test is used to determine the error on the fitted parameters. It takes the parameters as its only argument.
* **multi_gaussian_test** - A test function which produces test spectra with up to 4 Gaussian components. This test is used to determine how well the code is at detecting the correct number of components. It takes the parameters as its only argument.

The general manner in which the code should be run is for the parameter file to be read in using **read_parameters**, followed by **make_moments** to generate the moment maps, followed by **fit_a_fits** to fit the PPV datacube. This is because the mask generated to calculate accurate moments is also used by the spectral fitting routine. However, once the moment-making function has been completed the mask is saved as a fits files, and thus this step does not need to be run every time the fitting routine is called. If a single spectrum is fitted, i.e. using **fit_single_line**, then the mask must be generated by the user. The outputs for making moments and fitting fits files, as well as their format, is explained in the next section which details the methods employed in each function. 

The user interacts with BTS predominately via a parameter file which contains all important information. Included in the repository is an example script which shows how the 6 functions are used as well as example parameter files. An explanation of each of the parameters can be found at towards the end of this documentation. 

## Moment-masking technique

The three moments, 0,1 and 2, are commonly the first data products generated from PPV data cubes of molecular lines. However, it is known that noise in the spectra can affect the resulting moment maps. The typical technique used to combat this is called clipping, where only voxels in the PPV cube which have an intensity above some threshold (normally some multiple of the noise level, e.g. 5&sigma;) are included. Thus this technique minimises the effect of noise by ignoring low intensity voxels; however, by doing so it also ignores numerous voxels which do contain true emission and so biases the resulting moment maps. [Dame (2011)](http://arxiv.org/abs/1101.1499) go into detail to all of the negative effects of clipping and suggest a moment-masking technique which helps alleviate them.

BTS uses such a moment-masking technique, though slightly modified, to better capture all emission voxels to produce accurate moment maps. In addition the mask generated during the process is highly useful for the spectral fitting as it allows noise-only velocity channels to be excluded from fitting metrics and thus biasing the fitting process. 

The general outline of the moment-masking technique is this: the PPV cube is smoothed using a 3D top-hat filter of size (m,m,m); noise is estimated from this smoothed datacube on a pixel-by-pixel basis; voxels in the smoothed cube which have an intensity above some value T<sub>C</sub> are unmasked, i.e. are said to contain emission; voxels which neighbour previously unmasked voxels and also have an intensity above some value T<sub>L</sub>, where T<sub>L</sub> < T<sub>C</sub>, are unmasked; this previous step continues in an iterative manner until no new voxels are unmasked, completing the masking process; the voxels in the original PPV cube which were unmasked in the smoothed datacube are then used to calculate the moments. An example of the comparison between clipping and moment-masking is shown in figure 1, where m=3, T<sub>C</sub> = 8&sigma; and T<sub>L</sub> = 4&sigma;. Moment-masking is significantly better at producing accurate moment maps, where noise effects are minimised, while also capturing the extended, weaker emission. 

Figure 1: 
![alt text](https://github.com/SeamusClarke/BTS/blob/master/Images/moment_test.png "Figure 1")

From testing, the size of the top-hat filter should be kept relatively small, with m=3-5 found to be adequate. Optimal values for T<sub>C</sub> and T<sub>L</sub> should be found by the user; lower values mean noise may begin to effect the moment calculations while higher values exclude weaker emission. Experimentation finds that T<sub>C</sub> = 6-8&sigma; and T<sub>L</sub> = 3-5&sigma; typically work well. If no values are specified in the parameter file, BTS defaults to m=3, T<sub>C</sub> = 8&sigma; and T<sub>L</sub> = 4&sigma;.

Using the above technique, the **make_moments** function produces the moment 0, 1 and 2 maps and outputs them as 2D arrays stored in fits files, using the header information in the 3D PPV datacube provided, a 2D noise map stored as a fits file, as well as the 3D mask generated during the process stored as a fits file in the same format as the PPV datacube. Note that when calculating a moment 1 or 2 value for a pixel, **make_moments** requires that there are at least 4 channels of emission in the spectrum; thus, there may exist pixels where a moment 0 is calculated which do not have a corresponding moment 1 or 2 due to very low signal-to-noise.

Note that there is also a limited velocity range mode for the moment calculations, i.e. if one wishes to calculate moment 0, 1 and 2 from a cube over only a certain velocity range. This is done by turning the parameter flag **use_velocity_range** to 1 and then specifying the desired velocity range with the parameters **min_velocity_range** and **max_velocity_range**. One must ensure that the specified velocities are in the same unit as the velocity axis information in the fits file header. If the header uses frequency instead of velocity, BTS will automatically use km/s and use the header RESTFRQ keyword to make the conversion between frequency and velocity.  

## Multi-component fitting methodology 

The BTS routine does not assume the number of components in a spectrum _a priori_, but uses the first, second and third derivatives to determine the number and positions of the velocity components. A least-squared fitting routine is then used to determine the best fit with that number of components, checking for over-fitting.

Figure 2 shows the first three derivatives for a perfect Gaussian with a mean of zero and a standard deviation of one. One sees that at the location of the Gaussian's maximum: the first derivative is at zero and decreasing, there exists a local minimum in the second derivative, and the third derivative is at zero and is increasing. It is the local minimum in the second derivative which is used as the main indicator of a component, the first and third derivatives are used as additional checks. This is because the second derivative is more sensitive than the first derivative, but not as sensitive as the third derivative to small oscillations, and so can be used for the detection of shoulders. This is seen in figure 3 where a second Gaussian with the same width, and half the amplitude of the first, is placed at _x_ = 2. The first derivative is no longer zero at this location, but the second derivative shows a local minimum at x ~ 2.2 and the third derivative is zero. 

Figure 2: 
![alt text](https://github.com/SeamusClarke/BTS/blob/master/Images/test_gauss.png "Figure 2")
Figure 3:
![alt text](https://github.com/SeamusClarke/BTS/blob/master/Images/test_two_gauss.png "Figure 3")

Observed spectra and spectra from synthetic observations have noise, this will add considerable noise to the derivatives making the detection of a peak more difficult. Figure 4 shows the same Gaussian as seen in figure 1 but with noise added. The noise is obtained by sampling from a normal distribution with a mean of zero and a standard deviation of 0.04, leading to a peak signal-to-noise ratio of ~10. One can see that the second derivative is purely noise and shows no pronounced local minimum at the location of the peak. To combat this, one may smooth noisy spectra before determining the derivatives. To smooth the spectrum it is convolved with a Gaussian kernel with a smoothing length, _h_. If the smoothing length is too large it will over-smooth the spectrum and features will be erased (i.e. shoulders). If the smoothing length is too small then the derivatives will retain a high level of noise and too many peaks will be detected. The smoothed spectrum seen in figure 4 uses a smoothing length of 3 velocity channels, one can see that the second derivative is still noisy and there exist numerous local minima which are not associated with the location of the Gaussian's peak. To avoid classifying such minima in the second derivative as a separate velocity component, the intensity of the spectrum in these velocity channels is compared with a signal-to-noise threshold. Both the smoothing length and the signal-to-noise threshold are important free parameters which are left for the user to determine. The defaults are _h_ = 3 and a signal-to-noise ratio threshold of 5.

Figure 4
![alt text](https://github.com/SeamusClarke/BTS/blob/master/Images/noisy_gauss.png "Figure 4")

The locations of the local minima in the second derivative identify the number of, and positions of, the velocity components; these can be used as initial guesses for the velocity centroids of the velocity components. The intensity of the spectrum at the positions of the second derivative's minima are used as initial guesses for the amplitudes of the velocity components. To determine a guess for the width, the routine uses the full width half maximum (FWHM) of the peaks. The positions where the spectrum is equal to the half maximum is recorded and the difference is taken to be the FWHM. This is divided by the number of components detected between these velocities and then converted into a standard deviation assuming a perfect Gaussian.

These guesses (the number of components, their amplitudes, positions and widths) are given to a least-squared fitting routine as initial fit parameters. BTS uses the function **curve_fit**, found in the Scipy. This function is run using the Trust Region Reflective algorithm option as this allows bounds to be placed on the fitting parameters. The bounds used for the amplitude, _A_, the velocity centroid, _v_<sub>cent</sub>, and the width, &sigma;<sub>width</sub>, are:

* _n_ &sigma;<sub>noise</sub> <  _A_ < 2 x max(spectrum);
* _v_<sub>min</sub> < _v_<sub>cent</sub> < _v_<sub>max</sub>;
*  &sigma;<sub>min</sub> < &sigma;<sub>width</sub> < &sigma;<sub>max</sub>

where _n_ is a user defined multiple of the noise level, &sigma;<sub>noise</sub>, max(spectrum) is the maximum intensity of the spectrum,_v_<sub>min</sub> and _v_<sub>max</sub> are respectively the minimum and maximum velocity in the spectrum, &sigma;<sub>min</sub> and &sigma;<sub>max</sub> are user defined limits for the minimum and maximum component widths allowed. **curve_fit** returns the parameters of the best fit, and the estimated co-variance matrix for these parameters. There are cases when **curve_fit** will not be able to converge on a best fit, either because the initial guesses for the fit parameters were poor, or the &chi;<sup>2</sup> landscape is complicated. In these cases the lack of convergence is noted and no fit is recorded; however, in real data the rate of no convergence is very low, << 1%. 

Once a fit is found, the corrected Akaike Information Criterion (AICc) is calculated. Next, if the number of components fitted is below some user-defined maximum value, an additional component is added at the location where the residual is largest and a new secondary fit is conducted. If the number of components fitted is greater than 1, the weakest component which is closest to another component is removed and a new tertiary fit is conducted. At this point, there are at most 3 fits to the spectrum and the AICc of each fit is compared with each other. The fit with the smallest AICc is taken as the best fit as long as the difference is greater than the user-defined parameter **delta_AIC_limit**. If the differences in AICc is below **delta_AIC_limit**, then the fit with the fewest number of components is selected. This allows the number of components to be varied in case the initial guess was wrong. If the number of components changed when the comparison is made, the code will iterate and repeat this addition/removal of a component until the number of components stops varying, or some iteration limit is reached. Due to this, high values of **delta_AIC_limit** will act to guard against spurious components due to noise being added in the fitting procedure but may exclude weaker, true components, while the reverse is true for low values of **delta_AIC_limit**. Testing the code, we find that values in the range of 5-20 seem reasonable, with **delta_AIC_limit** = 10 being the default value.

When all of the pixels in the PPV cube are fitted, 4 fits files are produced: Coeff, which contains the co-efficients of all of the fitted velocity components, formatted using the same spatial grid as the PPV cube; Error, which contains the errors on the model co-efficients; Model, which has the same shape as the inputted PPV cube and contains the noise-free model spectra resulting from the fits; and Res, which stores the residuals divided by the noise. Both the Coeff and Error files contain 4D arrays: the first dimension is always size 3, where the first element is the fitted amplitude, the second the fitted centroid, and the third the fitted width; the second dimension has a size equal to the maximum number of velocity components specified by the user, i.e. **max_peaks**, where each element stores the fitting parameters for the corresponding velocity component; the third and fourth dimensions are the same size as the spatial dimensions of the fitted cube. Entries in the 4D arrays which are equal to 0.0 mean that no corresponding velocity component is fitted; e.g. if co_eff[0,0,i,j] > 0 and co_eff[0,1,i,j] == 0 then in pixel *ij* only one velocity component is fitted.

### The parameter file

There are 7 sets of parameters in the parameter file: the important three, input/output names, spectral fitting parameters, noise level parameters, moment-masking parameters, flags, and test parameters. The example parameter files show typical/default values of the parameters and is well commented.

#### The important three
These are the three important parameters discussed in depth in the methodology section.

* **delta_AIC_limit** - This is the difference in AICc which is required for more complicated models to be selected (default 10.0). 
* **smoothing_length** - This is the smoothing length used to smooth the noisy spectrum before the determination of the number of peaks (default 3.0).
* **signal_to_noise_ratio** - This is the signal to noise ratio used to determine if a component is significant enough to be fitted (default 5.0).

#### Input/Output names

These are the two parameters which control the names of the input and output fits files.

* **data_in_file_name** - This is the name and location of the fits file one wishes to use to make moment maps from/fit spectra.
* **output_base** - This is the name base for the output fits file, i.e. Coeff_base.fits, Model_base.fits, base_mom0.fits

#### Spectral fitting parameters

These parameters are used when fitting spectra.

* **max_peaks** - The maximum number of velocity components to be considered when fitting spectra (default 3).
* **max_iterations** - The maximum number of iterations allowed when fitting spectra (default 5).
* **min_velocity_channels** - The minimum number of velocity channels which are considered to be emission for BTS to attempt to fit the spectrum (default 3).
* **min_width_value** - The minimum standard deviation that a velocity component may have; a good value is between 0.5-1.0 times the velocity resolution of the spectra. Will be in units of m/s or km/s depending on the unit used in the PPV cube's header (default 0.1).
* **max_width_value** - The maximum standard deviation that a velocity component may have. Will be in units of m/s or km/s depending on the unit used in the PPV cube's header (default 20.0).
* **mask_pad** - The number of velocity channels which are unmasked around emission channels before fitting, helps produce accurate component widths (default 2).

#### Noise level parameters

These are the parameters which tell the code the noise level of the spectra it is attempting to fit.

* **variable_noise** - In observations the noise level may vary across the map and so an assumption of a constant noise is not a good one. This is set to 1 if one wishes to have a noise which varies pixel-by-pixel, and set to 0 if a constant level is used (default 0).
* **noise_level** - This is the noise level that is used by the fitting routine for all spectra if variable_noise=0 (default 0.1).

#### Moment-masking parameters

These parameters are related to the moment-masking technique to generate moment and noise maps, as well as the emission mask.

* **velocity_channel_number** - If variable_noise=1, the standard deviation of the first N velocity channels, where N is equal to velocity_channel_number, is used to estimate the noise. Ensure these channels are truly emission free. A negative value may be given and this means that the last N velocity channels are used (default 30).
* **upper_sigma_level** - The upper threshold used in moment-masking, T<sub>C</sub>, expressed as a multiple of noise, so upper_sigma_level=8 means T<sub>C</sub> = 8&sigma; (default 8).
* **lower_sigma_level** - The lower threshold used in moment-masking, T<sub>L</sub>, expressed as a multiple of noise, so lower_sigma_level=4 means T<sub>L</sub> = 4&sigma; (default 4).
* **mask_filter_size** - The size of the 3D top-hat filter used when smoothing before making the mask (default 3).
* **use_velocity_range** - A flag to determine if the moments will be calculated over only a limited velocity range; set to 1 to use and set to 0 to not use (default 0).
* **min_velocity_range** - The minimum velocity in the range over which the moments will be calculated if the limited velocity range mode is activated (default -10).
* **max_velocity_range** - The maximum velocity in the range over which the moments will be calculated if the limited velocity range mode is activated (default 10).

#### Flags

These is only one flag used in the code

* **debug** - A debug flag which turns on a lot of screen output. Useful if things are going wrong.

#### Test run parameters

These parameters are used when one uses the single or multiple Gaussian test functions.

* **test_number** - The number of test spectra used.
* **test_spec_min** - The minimum velocity of the spectra.
* **test_spec_max** - The maximum velocity of the spectra.
* **test_spec_num** - The number of velocity channels in the spectra. 
* **test_noise** - TThe noise level added to the test spectra.
* **test_amplitude_min** - The minimum amplitude of the components in the test spectra.
* **test_amplitude_max** - The maximum amplitude of the components in the test spectra.
* **test_width_min** - The minimum width of the components in the test spectra.
* **test_width_max** - The maximum width of the components in the test spectra.
* **test_vel_cen_min** - The minimum velocity centroid of the components in the test spectra.
* **test_vel_cen_max** - The maximum velocity centroid of the components in the test spectra.
* **test_plot_tag** - A flag to turn on plotting for the single_gaussian_test function.


### Run tests to determine parameter values

Finally we discuss the two test routines included in BTS; these may be used by the user to help determine optimal values for the main three parameter: **delta_AIC_limit**, **smoothing_length** and **signal_to_noise_ratio**. 

The first test routine, **single_gaussian_test**, fits test spectra constructed with only a single velocity component to check the accuracy of the fitting parameters; an example parameter file is included for this purpose and shows typical median errors of ~0.5-1%. Further, in the example test with 2000 test spectra, BTS never fits more than one velocity component even when **max_peaks** is set to 4. 

The second test routine, **multi_gaussian_test**, fits test spectra which consists of between 1 and 4 velocity components to check the accuracy of the number of components selected by BTS; an example parameter file is also included for this purpose. With the example parameters, after nearly 2000 tests, BTS never selects a model with the incorrect number of velocity components.

## Acknowledgements 
Version 1.0 of this code was produced with the support of the ERC starting grant No. 679852 "RADFEEDBACK". Version 2.0 of this code was produced with the support of the Ministry of Science and Technology (MoST) in Taiwan through grant MoST 108-2112-M-001-004-MY2.

## Contact

For up-to-date contact information see my [personal website](https://seamusclarke.github.io/#five)
