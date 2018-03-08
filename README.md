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

## User guide

Here we discuss the main functions which the user may call from the module. 

The module is contained and there are only 4 functions which the user will call:

* fit_single_line,
* fit_a_fits,
* single_gaussian_test,
* multi_gaussian_test.
