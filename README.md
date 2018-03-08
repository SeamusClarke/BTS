# The BrillianT Spectral fitting code

The BrillianT Spectral (BTS) fitting code is a python module designed to be a fully-automated multiple-component fitter for optically-thin spectra. The code is open-source and can be downloaded here. We ask that if the code is used in a publication that the Clarke et al. 2018 paper, which shows the first use, a description and tests of the code, is cited.

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

## Algorithm methodology 

The BTS routine does not assume the number of components in a spectrum _a priori_, but uses the first, second and third derivatives to determine the number and positions of the velocity components. A least-squared fitting routine is then used to determine the best fit with that number of components, checking for over-fitting and over-lapping velocity centroids.

The figure 1 shows the first three derivatives for a perfect Gaussian with a mean of zero and a standard deviation of one. One sees that at the location of the Gaussian's maximum: the first derivative is at zero and decreasing, there exists a local minimum in the second derivative, and the third derivative is at zero and is increasing. It is the local minimum in the second derivative which is used as the main indicator of a component, the first and third derivatives are used as additional checks. This is because the second derivative is more sensitive than the first derivative, but not as sensitive as the third derivative to small oscillations, and so can be used for the detection of shoulders. This is seen in figure 2 where a second Gaussian with the same width, and half the amplitude of the first, is placed at _x_ = 2. The first derivative is no longer zero at this location, but the second derivative shows a local minimum at x ~ 2.2 and the third derivative is zero. 

Figure 1: 
![alt text](https://github.com/SeamusClarke/BTS/blob/master/Images/test_gauss.png "Figure 1")
Figure 2:
![alt text](https://github.com/SeamusClarke/BTS/blob/master/Images/test_two_gauss.png "Figure 2")

Observed spectra and spectra from synthetic observations have noise, this will add considerable noise to the derivatives making the detection of a peak more difficult. Figure 3 shows the same Gaussian as seen in figure 1 but with noise added. The noise is obtained by sampling from a Gaussian distribution with a mean of zero and a standard deviation of 0.04, leading to a peak signal-to-noise ratio of ~ 10. One can see that the second derivative is purely noise and shows no pronounced local minimum at the location of the peak. To combat this, one may smooth noisy spectra before determining the derivatives. To smooth the spectrum it is convolved with a Gaussian kernel with a smoothing length, _h_. If the smoothing length is too large it will over-smooth the spectrum and features will be erased (i.e. shoulders). If the smoothing length is too small then the derivatives will retain a high level of noise and too many peaks will be detected. The smoothed spectrum seen in figure 3 uses a smoothing length of 3 velocity channels, one can see that the second derivative is still noisy and there exist numerous local minima which are not associated with the location of the Gaussian's peak. To avoid classifying such minima in the second derivative as a separate velocity component, the intensity of the spectrum in these velocity channels is compared with a signal-to-noise threshold. Both the smoothing length and the signal-to-noise threshold are important free parameters which are left for the user to determine. The defaults are _h_ = 3 and a signal-to-noise ratio threshold of 5.

Figure 3
![alt text](https://github.com/SeamusClarke/BTS/blob/master/Images/noisy_gauss.png "Figure 3")

The locations of the local minima in the second derivative identify the number of, and positions of, the velocity components; these can be used as initial guesses for the velocity centroids of the velocity components. The intensity of the spectrum at the positions of the second derivatives's minima are used as initial guesses for the amplitudes of the velocity components. To determine a guess for the width, the routine uses the full width half maximum (FWHM) of the peaks. The positions where the spectrum is equal to the half maximum is recorded and the difference is taken to be the FWHM. This is divided by the number of components detected between these velocities and then converted into a standard deviation assuming a perfect Gaussian.

These guesses  (the number of components, their amplitudes, positions and widths) are given to a least-squared fitting routine as initial fit parameters. BTS uses the function **curve_fit**, found in the Scipy. This function is run using the Trust Region Reflective algorithm option as this allows bounds to be placed on the fitting parameters. The bounds used for the amplitude, _A_, the velocity centroid, _v_<sub>cent</sub>, and the width, &sigma;<sub>width</sub>, are:

* _n_ &sigma;<sub>noise</sub> <  _A_ < 2 x max(spectrum);
* _v_<sub>min</sub> < _v_<sub>cent</sub> < _v_<sub>max</sub>;
* &delta;v < &sigma;<sub>width</sub> < _v_<sub>max</sub> - _v_<sub>min</sub>

where _n_ is a user defined multiple of the noise level, &sigma;<sub>noise</sub>, max(spectrum) is the maximum intensity of the spectrum,_v_<sub>min</sub> and _v_<sub>max</sub> are respectively the minimum and maximum velocity in the spectrum, and &delta;v is the spectral resolution. **curve_fit** returns the parameters of the best fit, and the estimated covariance matrix for these parameters. There are cases when **curve_fit** will not be able to converge on a best fit, either because the initial guesses for the fit parameters were poor, or the &chi;<sup>2</sup> landscape is complicated. In these cases the lack of convergence is noted and no fit is recorded; however, in real data the rate of no convergence is very low, << 1%. In the future, a Monte Carlo Markov Chain (MCMC) fitting routine will be added to fit those spectra which cannot be fitted using **curve_fit**. 

Once a fit is found, the reduced &chi;<sup>2</sup> is calculated. The reduced &chi;<sup>2</sup> is compared to a user defined limit, &chi;<sup>2</sup><sub>limit</sub>. Those fits which return a value above this limit are refitted with an additional velocity component added. The initial guess for the new component's centroid is taken to be the velocity of the channel at which the absolute residual is largest. The initial guess for amplitude is then taken to be the intensity of the spectra at this location and the width guess is taken to be the velocity resolution. If the new fit lies below the &chi;<sup>2</sup> limit then the new set of fitting parameters is saved, if they do not then the fitting parameters for the old fit are recorded. To avoid over-fitting, those fits which have reduced &chi;<sup>2</sup> below the limit, &chi;<sup>2</sup><sub>limit</sub>, are re-fitted using one fewer velocity component. The removed velocity component is the one with the smallest amplitude. If the fit with fewer velocity components also has a reduced &chi;<sup>2</sup> below &chi;<sup>2</sup><sub>limit</sub>, then its fitting parameters are saved; if not, the old fit is saved. The default valueis &chi;<sup>2</sup><sub>limit</sub> = 1.5. 

As well as checking for over-fitting, the code checks for over-lapping velocity components. Such components may appear in spectra for physical reasons (i.e. jets) and so this feature may be disabled. However, if over-lapping velocity components are not desired the code checks if any two of the returned velocity centroids lie within one velocity channel of each other. If there are such over-lapping components then the weaker of the two over-lapping components is removed and the fit repeated. 


## User guide

Here we discuss the main functions which the user may call from the module. 

The module is contained and there are only 4 functions which the user will call:

* **fit_single_line**,
* **fit_a_fits**,
* **single_gaussian_test**,
* **multi_gaussian_test**.
