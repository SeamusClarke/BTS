import numpy
import matplotlib.pyplot
from scipy.optimize import curve_fit
from astropy.convolution import convolve, Gaussian1DKernel, convolve_fft
import astropy.io.fits
import os

############ Fitting routine itself

def fit_single_line(vel,x,mask,params):

	####### Unpack the parameter array
	debug = params["debug"]
	smooth = params["smoothing_length"]
	var_noise = params["variable_noise"]
	noise_level = params["noise_level"]
	n = params["signal_to_noise_ratio"]
	max_peaks = params["max_peaks"]
	max_it = params["max_iterations"]
	daic = params["delta_AIC_limit"]
	min_num_channels = params["min_velocity_channels"]
	min_vw = params["min_width_value"]
	max_vw = params["max_width_value"]
	mask_pad = params["mask_pad"]

	### If the spectrum mask shows fewer than the minimum number of velocity channels of emission then skip spectrum fitting
	if(sum(mask)<min_num_channels):
		return [[-2,0,0],[0,0,0],1e9]


	if(debug==1):
		print( "##########")
		print( "DEBUG MODE")
		print( "##########")
		print( " ")

	### Get info about velocity
	nv = len(x)
	maxv=max(vel)
	minv=min(vel)
	dv = numpy.fabs(vel[1] - vel[0])

	####### Determine the noise level
	if(var_noise == 1):
		noise = numpy.std(x[mask==0])
	else:
		noise = noise_level
		

	### Pad the mask so that a some noise channels on either side of the emission channels are still included in all the calculations
	mask = pad_mask(mask,mask_pad)

	#### prepare the data and convolve spectrum with Gaussian for peak determination.
	spec = x[:]
	gk = Gaussian1DKernel(smooth)
	smo_spec = convolve(spec,gk)

	### Work out the gradients of the spectrum
	dspec = numpy.zeros_like(spec)
	for ii in range(0,nv-1):
		dspec[ii] = (smo_spec[ii+1]-smo_spec[ii])/dv

	ddspec = numpy.zeros_like(spec)
	for ii in range(0,nv-2):
		ddspec[ii] = (dspec[ii+1]-dspec[ii])/dv

	dddspec = numpy.zeros_like(spec)
	for ii in range(0,nv-3):
		dddspec[ii] = (ddspec[ii+1]-ddspec[ii])/dv

	switch = numpy.zeros_like(dspec)

	decrease = 0

	### go through and work out the number of peaks

	for ii in range(0,nv-2):

			if(ddspec[ii+1] < ddspec[ii] and ddspec[ii+2] < ddspec[ii]):
				decrease = 1
			if(ddspec[ii+1] > ddspec[ii] and ddspec[ii+2] > ddspec[ii] and dddspec[ii]>0):
				if(decrease==1 and (mask[ii]==1 or mask[ii+1]==1 or mask[ii-1]==1) and (spec[ii]> n*noise or spec[ii+1]> n*noise or spec[ii-1]> n*noise) ):	
					switch[ii] = 1
				decrease = 0

	### if there seems to be no peaks, something went wrong so just make a guess of a single component
	if(sum(switch)<1):
		guess = single_gauss_guess(vel,spec,mask)
		### Maybe a bad pixel/too few emission channels even after padding so return non-convergence tag
		if(numpy.isnan(guess[2]) or guess[0]<n*noise):
			return [[-3,0,0],[0,0,0],1e9]
		### If guess is ok then proceed
		n_peaks = 1
		pid=1
		bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
		bound[0][0] = n*noise
		bound[1][0] = 2*max(spec) + n*noise
		bound[0][1] = minv
		bound[1][1] = maxv
		bound[0][2] = min_vw
		bound[1][2] = max_vw

		if(guess[0]<n*noise):
			guess[0]=1.01*n*noise
		if(guess[1]>maxv):
			guess[1]=maxv-dv
		if(guess[1]<minv):
			guess[1]=minv+dv
		if(guess[2]>max_vw):
			guess[2] = 0.99*max_vw
		if(guess[2]<min_vw):
			guess[2]=1.01*min_vw


	else:
		### here we set up the arrays that contain the guesses for the peaks' amplitudes and centriods and widths
		index = numpy.linspace(0,nv-1,nv)
		index = numpy.array(index,dtype=numpy.int)

		pid = index[switch==1] + 1
		pcent = vel[pid]
		pamp = spec[pid]

		pcent = numpy.array(pcent,dtype=numpy.double)
		pamp = numpy.array(pamp,dtype=numpy.double)

		high = numpy.zeros_like(spec)
		high = numpy.array(high,dtype=numpy.int)

		high[spec>0.5*min(pamp)] = numpy.int(1)

		start=-1
		finish=-1

		psig = numpy.zeros(len(pid))

		num=0

		### Go through and work out a guess for the widths

		for ii in range(0,nv-1):

			if(ii==0 and high[0]==1):
				start=ii

			if(high[ii+1] == 1 and high[ii] == 0 and start==-1 and finish==-1):
				start=ii

			if(high[ii+1] == 0 and high[ii] == 1 and start!=-1):
				finish = ii

				ran = numpy.fabs(vel[finish] -vel[start])

				l = len(pid[(pid<finish)*(pid>start)])

				for jj in range(num,num+l):
					psig[jj] = ran/(l*numpy.sqrt(8*numpy.log(2)))

				num=num+l
				
				if(debug==1):
					print( "start  = ", vel[start])
					print( "finish = ", vel[finish])

				start=-1
				finish=-1


		### if a guess width is smaller than the velocity resolution then we set it to the velocity resoution
		psig[psig<min_vw] = 1.01 * min_vw

		### If more peaks are guessed than maximum allowed peaks then keep on the highest amplitude peaks up to the max_peak number
		n_peaks = len(pamp)
		if(n_peaks>max_peaks):
			if(debug==1):
				print("More than", max_peaks,"peaks were found")
			dum = numpy.argsort(pamp)
			pamp = pamp[dum[-max_peaks:]]
			psig = psig[dum[-max_peaks:]]
			pcent = pcent[dum[-max_peaks:]]
			pid = pid[dum[-max_peaks:]]
			n_peaks=max_peaks

		### set limits on the guess and fill the guess and boundary arrays

		guess = numpy.zeros(3*n_peaks)
		bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
		guess = numpy.array(guess,dtype=numpy.double)
			
		for ii in range(0,n_peaks):

			if(pamp[ii]<n*noise):
				pamp[ii]=1.01*n*noise
			if(pcent[ii]>maxv):
				pcent[ii]=maxv-dv
			if(pcent[ii]<minv):
				pcent[ii]=minv+dv
			if(psig[ii]>max_vw):
				psig[ii] = 0.99*max_vw
			if(psig[ii]<min_vw):
				psig[ii]=1.01*min_vw

			guess[3*ii] = pamp[ii]
			guess[3*ii+1] = pcent[ii]
			guess[3*ii+2] = psig[ii]

			bound[0][3*ii] = n*noise
			bound[1][3*ii] = 2*max(spec) + n*noise
			bound[0][3*ii+1] = minv
			bound[1][3*ii+1] = maxv
			bound[0][3*ii+2] = min_vw
			bound[1][3*ii+2] = max_vw


	if(debug==1):
		print( "######## Guess values #########")
		print( "Number of peaks = ", n_peaks)
		print( "Peak ids = ", pid)
		print( "Peak centroids = ", guess[1::3])
		print( "Peak amplitude = ", guess[::3])
		print( "Peak width = ", guess[2::3])
		print( " ")



	### Fit for the first time using the initial guesses
	nit=0
	co_eff, errors, converged, model, AIC = fit_guess(vel,spec,mask,guess,bound,noise)
	keep_going=1

	### Debug display information and if no convergence is reached then try a single component guess and fit again
	if(debug==1):
		print("##### After first fit #####")
		print("Co_eff = ", co_eff)
		print("AIC = ", AIC)
		print(" ")

	### No convergence is found after the first fit, try to simplify and try again with a single component
	if(converged!=1):
		if(debug==1):
			print("First fit did not converge")
			print("Guess parameters = ",guess)
			print("Noise = %.3lf, velocity channel = %.3lf" %(noise, dv))

		guess = single_gauss_guess(vel,spec,mask)
		### Maybe a bad pixel/too few emission channels even after padding so return non-convergence tag
		if(numpy.isnan(guess[2]) or guess[0]<n*noise):
			return [[-3,0,0],[0,0,0],1e9]

		### Else proceed
		n_peaks = 1
		bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
		bound[0][0] = n*noise
		bound[1][0] = 2*max(spec) + n*noise
		bound[0][1] = minv
		bound[1][1] = maxv
		bound[0][2] = min_vw
		bound[1][2] = max_vw

		if(guess[0]<n*noise):
			guess[0]=1.05*n*noise
		if(guess[1]>maxv):
			guess[1]=maxv-dv
		if(guess[1]<minv):
			guess[1]=minv+dv
		if(guess[2]>max_vw):
			guess[2] = 0.99*max_vw
		if(guess[2]<min_vw):
			guess[2]=1.05*min_vw

		### Try a fit with this new guess
		co_eff, errors, converged, model, AIC = fit_guess(vel,spec,mask,guess,bound,noise)

		### If still not converging then exit for this spectrum
		if(converged!=1):
			print("Fit did not converge")
			return [[-1,0,0],[0,0,0],1e9]

		


	### Until the AIC minimum is found or a sufficient number of iterations are performed keep adding and removing components to find optimal solution

	while(keep_going==1 and nit<max_it and converged==1):

		### Reset the AICs and co_effs
		AIC_m = 1e9
		AIC_l = 1e9
		co_eff_m = co_eff
		co_eff_l = co_eff
		check=0

		#### Check for boundary components
		check = check_for_boundary(co_eff,bound)
		if(check==1):
			### If there is a boundary component, remove it and get new guesses without it
			guess_new, bound_new = remove_boundary_components(co_eff,bound)
			### If the new guess was reasonable fit using it
			if(guess_new[0]>0):
				co_eff_n,errors_n,converged_n,model_n,AIC_n = fit_guess(vel,spec,mask,guess_new,bound_new,noise)
				### If the fit using the new guess without the boundary component is converged we take it
				if(converged_n==1):
					n_peaks = int(len(co_eff_n[::3]))
					co_eff = co_eff_n
					errors = errors_n
					converged = converged_n
					model = model_n
					AIC = AIC_n
					guess = guess_new
					bound = bound_new

		### Add an additional component if the maximum number of peaks hasn't been reached and fit again
		if(n_peaks<max_peaks):
			guess_m, bound_m = add_component(vel,spec,mask,model,co_eff,bound)
			co_eff_m, errors_m, converged_m, model_m, AIC_m = fit_guess(vel,spec,mask,guess_m,bound_m,noise)
		else:
			AIC_m = 1e9

		### If there are currently more than 1 peaks, remove one of the peaks and fit again
		if(n_peaks>1):
			guess_l, bound_l = remove_component(vel,spec,mask,model,co_eff,bound)
			co_eff_l, errors_l, converged_l, model_l, AIC_l = fit_guess(vel,spec,mask,guess_l,bound_l,noise)
		else:
			AIC_l = 1e9
		
		### Debug info
		if(debug==1):
			print("##### After %d fit #####" %(nit+1))
			print("Keep going check = %d, Number of peaks = %d" %(keep_going, n_peaks))
			print("AIC of npeaks = %.3lf, AIC of npeaks-1 = %.3lf, AIC of npeaks+1 = %.3lf" %(AIC,AIC_l,AIC_m))
			print("Co_eff of npeaks:", co_eff)
			print("Co_eff of npeaks-1:", co_eff_l)
			print("Co_eff of npeaks+1:", co_eff_m)
			print(" ")


		### Current fit has a better AIC than both the added and removed component fits, thus we stop the iterations
		if((AIC + daic < AIC_l and AIC <= AIC_m + daic)):
			keep_going=0

		### Removing a component leads to better fit than both the added component and current fits. Repeat again with this new fit
		if(AIC_l <= AIC + daic):
			co_eff = co_eff_l
			errors = errors_l
			converged = converged_l
			model = model_l
			AIC = AIC_l
			guess = guess_l
			bound = bound_l
			n_peaks = n_peaks - 1
			keep_going = 1

		### Adding a component leads to a better fit than both the current and removed component fits. Repeat again with this new fit
		if(AIC_m + daic < AIC and AIC_l>AIC+daic):
			co_eff = co_eff_m
			errors = errors_m
			converged = converged_m
			model = model_m
			AIC = AIC_m
			guess = guess_m
			bound = bound_m
			n_peaks = n_peaks + 1
			keep_going = 1

		### Add an iteration to the counter
		nit = nit+1
		
	### If there is no convergence at any point return error
	if(converged!=1):
		print("Fit did not converge")
		return [[-1,0,0],[0,0,0],1e9]

	### If the number of iterations is too high then return best fit so far and inform user
	if(nit==max_it):
		#print("Took lots of iterations")
		return co_eff, errors, AIC


	return co_eff, errors, AIC

### Function to pad the mask around the emission channels
def pad_mask(mask,mask_pad):

	nv = len(mask)
	ii = mask_pad
	while(ii<nv-mask_pad):
		if(mask[ii]==1 and mask[ii-1]==0):
			mask[ii-mask_pad:ii] = 1
			ii = ii + 1
			continue
		elif(mask[ii]==0 and mask[ii-1]==1):
			mask[ii:ii+mask_pad] = 1
			ii = ii + mask_pad + 1
			continue
		else:
			ii = ii + 1

	return mask


### A function to determine a single component guess to help poor convergence spectra
def single_gauss_guess(vel,spec,mask):

	# Consider only parts of the spectra masked as emission
	vv = vel[mask==1]
	ss = spec[mask==1]
	# Discard channels with negative intensity values to avoid biasing moment estimates
	vv = vv[ss>0]
	ss = ss[ss>0]
	# Calculate moments 1 and 2 and use them as estimates of the centroid and width of the single component guess
	mom1, mom2 = weighted_avg_and_std(vv,ss)
	guess = numpy.zeros(3,dtype=numpy.double)
	dv = vel[1]-vel[0]
	mom1_index = int((mom1 - vel[0])/dv) + 1
	guess[0] = spec[mom1_index]
	guess[1] = mom1
	guess[2] = mom2
	return guess

### Function to return N superimposed Gaussian functions 
def multi_gauss(vel,params):
	N = int(len(params)/3)
	a,b,c = numpy.array(params[::3]),numpy.array(params[1::3]),numpy.array(params[2::3])
	y = numpy.zeros_like(vel)
	for ii in range(0,N):
		y = y + a[ii]*numpy.exp(-(vel-b[ii])**2 / (2*c[ii]**2))
	return y


### Function to fit a N Gaussian function
def fit_guess(vel,spec,mask,guess,bound,noise):

	try:
		co_eff, var_matrix = curve_fit(lambda vel, *guess: multi_gauss(vel,guess),vel,spec,p0=guess,method="trf",bounds=bound)
		errors = numpy.sqrt(numpy.diag(var_matrix))
		converged = 1
		model = multi_gauss(vel,co_eff)
		AIC = calc_AIC(vel,spec,co_eff,model,mask,noise)
		if(AIC==1e9):
			converged=0

	except RuntimeError:
		co_eff = numpy.zeros_like(guess)
		errors = numpy.zeros_like(guess)
		converged=0
		model = numpy.zeros_like(vel)
		AIC = 1e9

	return co_eff, errors, converged, model, AIC


### Calculate the corrected AIC using only the velocity channels with emission
def calc_AIC(vel,spec,co_eff,model,mask,noise):

	res = spec - model 
	res = res[mask==1]

	### Number of data points equal to number of velocity channels used to calculate residuals
	### Number of parameters of the model is equal to the number of parameters for the fitting
	n = len(res)
	k = int(len(co_eff))

	### If there are too few data points to calculate the correction factor then return extremely high AIC
	if(n-k-1 < 1):
		AIC=1e9
	else:
		AIC = 2*k + n*numpy.log(2*numpy.pi) + n*numpy.log(noise**2) + (sum(res**2))/(noise**2) + (2*k**2 + 2*k)/(n-k-1)

	return AIC


### Add a component by finding the location of the maximum residual
def add_component(vel,spec,mask,model,guess,bound):

	res = spec - model
	res[mask==0] = 0

	### Expand bound array first
	n_peaks = int(len(guess[::3]))
	bound2 = (numpy.zeros(3*n_peaks + 3),numpy.zeros(3*n_peaks + 3))

	for ii in range(0,n_peaks+1):

		bound2[0][3*ii] = bound[0][0]
		bound2[1][3*ii] = bound[1][0]
		bound2[0][3*ii+1] = bound[0][1]
		bound2[1][3*ii+1] = bound[1][1]
		bound2[0][3*ii+2] = bound[0][2]
		bound2[1][3*ii+2] = bound[1][2]

	### Now do guess
	guess2 = numpy.zeros(len(guess) + 3)
	guess2[:3*n_peaks] = guess[:3*n_peaks]

	### Find place with highest residual
	guess2[3*n_peaks] = numpy.amax(res)
	high_res = numpy.argmax(res)
	guess2[3*n_peaks+1] = vel[high_res]
	guess2[3*n_peaks+2] = 4.01*bound[0][2]

	### Do some double checks with the guessed parameters to ensure they lie within the bounds.
	if(guess2[3*n_peaks]<bound2[0][0]):
		guess2[3*n_peaks] = 1.5 * bound2[0][0]
	if(guess2[3*n_peaks]>bound2[1][0]):
		guess2[3*n_peaks] = 0.9 * bound2[1][0]
	
	return guess2,bound2

### Remove a component by removing the smallest amplitude component
def remove_component(vel,spec,mask,model,guess,bound):
	
	### Reduce bound array first
	n_peaks = int(len(guess[::3]))
	bound2 = (numpy.zeros(3*n_peaks - 3),numpy.zeros(3*n_peaks - 3))

	for ii in range(0,n_peaks-1):

		bound2[0][3*ii] = bound[0][0]
		bound2[1][3*ii] = bound[1][0]
		bound2[0][3*ii+1] = bound[0][1]
		bound2[1][3*ii+1] = bound[1][1]
		bound2[0][3*ii+2] = bound[0][2]
		bound2[1][3*ii+2] = bound[1][2]

	### Extract the three parameters of the Gaussian fits
	pamp = guess[::3]
	pcent = guess[1::3]
	psig = guess[2::3]

	### Find the two components which are closest together and pick the lower amplitude one.
	diff_v = 1e9
	min_diff_v = 0
	for ii in range(0,n_peaks):
		for jj in range(ii+1,n_peaks):

			ddv = numpy.fabs(pcent[ii] - pcent[jj])
			if(ddv<diff_v):
				diff_v = ddv
				if(pamp[ii]<pamp[jj]):
					min_diff_v=ii
				else:
					min_diff_v=jj

	guess2 = numpy.zeros(len(guess) - 3)
	ind = numpy.arange(0,n_peaks,1)

	guess2[::3] = pamp[ind!=min_diff_v]
	guess2[1::3] = pcent[ind!=min_diff_v]
	guess2[2::3] = psig[ind!=min_diff_v]

	return guess2,bound2


### Check if the co-efficients are too close to the minimum amplitude or width boundaries. Typical of bad fits
def check_for_boundary(co_eff,bound):

	q = numpy.sum(co_eff[::3] < 1.02*bound[0][0]) + numpy.sum(co_eff[2::3] < 1.02*bound[0][2])

	if(q>0):
		check_bound=1
	else:
		check_bound=0

	return check_bound

### Remove the components which are close to the minimum amplitude or width boundaries
def remove_boundary_components(co_eff,bound):

	#### First check the amplitudes
	q = numpy.sum(co_eff[::3] < 1.02*bound[0][0])

	### Check is every component has a tiny amplitude
	if(q==len(co_eff[::3])):
		### If there is only one component we can't remove it so return it
		if(q==1):
			return co_eff, bound
		### If there is more than one component then remove all but the first and try again	
		if(q>1):
			return co_eff[:3], [bound[0][:3],bound[1][:3]]

	### There are components with tiny amplitudes, but not all		
	if(q>0):
		### Make new bounds array
		n_peaks = len(co_eff[::3]) - q
		bound2 = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))

		for ii in range(0,n_peaks):
			bound2[0][3*ii] = bound[0][0]
			bound2[1][3*ii] = bound[1][0]
			bound2[0][3*ii+1] = bound[0][1]
			bound2[1][3*ii+1] = bound[1][1]
			bound2[0][3*ii+2] = bound[0][2]
			bound2[1][3*ii+2] = bound[1][2]

		### Copy over the components which have amplitudes above the minimum
		guess2 = numpy.zeros(3*n_peaks)
		p = co_eff[::3] > 1.02*bound[0][0]
		guess2[::3] = co_eff[::3][p]
		guess2[1::3] = co_eff[1::3][p]
		guess2[2::3] = co_eff[2::3][p]

		co_eff = guess2
		bound=bound2


	### Now check the widths
	q = numpy.sum(co_eff[2::3]<1.02*bound[0][2])

	### Check is every component has a tiny width
	if(q==len(co_eff[::3])):
		### If there is only one component we can't remove it so return it
		if(q==1):
			return co_eff, bound
		### If there is more than one component then remove all but the first and try again
		if(q>1):
			return co_eff[:3] , [bound[0][:3],bound[1][:3]]

	### There are components with tiny amplitudes, but not all
	if(q>0):

		### Make new bounds array
		n_peaks = len(co_eff[::3]) - q
		bound2 = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))

		for ii in range(0,n_peaks):
			bound2[0][3*ii] = bound[0][0]
			bound2[1][3*ii] = bound[1][0]
			bound2[0][3*ii+1] = bound[0][1]
			bound2[1][3*ii+1] = bound[1][1]
			bound2[0][3*ii+2] = bound[0][2]
			bound2[1][3*ii+2] = bound[1][2]

		### Copy over the components which have amplitudes above the minimum
		guess2 = numpy.zeros(3*n_peaks)
		p = co_eff[2::3] > 1.02*bound[0][2]
		guess2[::3] = co_eff[::3][p]
		guess2[1::3] = co_eff[1::3][p]
		guess2[2::3] = co_eff[2::3][p]

		co_eff = guess2
		bound=bound2

	return co_eff,bound










##### Test run functions, used to help determine the optimal values for the 3 important parameters: chi_limit, smoothing_length and signal to noise ratio

### The first is for the single Gaussian test, used to determine the average error on the fitting parameters

def single_gaussian_test(param):


	### Unpack the parameter array

	num_test = param["test_number"]
	
	spec_min = param["test_spec_min"]
	spec_max = param["test_spec_max"]
	spec_nv  = param["test_spec_num"]

	noise_level =  param["test_noise"]

	amp_min = param["test_amplitude_min"]
	amp_max = param["test_amplitude_max"]
	cen_min = param["test_vel_cen_min"]
	cen_max = param["test_vel_cen_max"]
	wid_min = param["test_width_min"]
	wid_max = param["test_width_max"]

	plot_tag = param["test_plot_tag"]

	var_noise = param["variable_noise"]
	if(var_noise!=0):
		print("variable_noise should be set to 0 for this test. Setting it to 0 now.")
		param["variable_noise"] = 0


	### Construct the velocity array

	v = numpy.linspace(spec_min,spec_max,spec_nv)

	### Initiize the seed for the random number generators
	numpy.random.seed(4)

	### Counters
	n=0
	n2=0

	### Empty lists to store the errors on the fitting parameters
	a_e = []
	c_e = []
	w_e = []
	c_chi = []

	### Loop over the number of tests specified

	print( "#######################")
	print( "#### Test progress ####")
	print( "#######################")
	print( " ")

	for ii in range(0,num_test):

		if((ii+1)%numpy.int(num_test/10.)==0):
			print( "The test is %.2f %% complete" %(100*(ii+1)/numpy.float(num_test)))


		### Pick a random amplitude, centroid and width
		amp = numpy.random.rand(1)*(amp_max - amp_min) + amp_min
		cen = numpy.random.rand(1)*(cen_max - cen_min) + cen_min
		width = numpy.random.rand(1)*(wid_max - wid_min) + wid_min

		### Produce Gaussian and add noise
		y = multi_gauss(v,[amp,cen,width])
		mask = numpy.zeros_like(y)
		mask[y>0.1*numpy.amax(y)] = 1
		no = numpy.random.normal(loc=0.0,scale=noise_level,size=spec_nv)
		y = y+no


		co_eff,errors,AIC =fit_single_line(v,y,mask,param)

		### Check that a fit was found
		if(co_eff[0]==-1):
			n = n + 1
			print( "No line was detected")
			print( "The amplitude, centroid and width was: ", amp, cen, width	)

		### If there was a fit, check for the number of components found
		else:

			### If only a single Gaussian found, store the errors on the parameters
			if(len(co_eff)==3):

				a_e.append(numpy.fabs(amp-co_eff[::3])/amp)
				c_e.append(numpy.fabs((cen-co_eff[1::3])/cen))
				w_e.append(numpy.fabs(width-co_eff[2::3])/width)


				### If the plot tag is turned on, plot the input parameter against the fitted parameters
				if(plot_tag==1):
					matplotlib.pyplot.figure(1)
					matplotlib.pyplot.plot(amp,co_eff[0],"kx")
					matplotlib.pyplot.figure(2)
					matplotlib.pyplot.plot(cen,co_eff[1],"kx")
					matplotlib.pyplot.figure(3)
					matplotlib.pyplot.plot(width,co_eff[2],"kx")


			### More than one component was detected.
			else:
				n2 = n2 + 1
				print( "More than one component was detected")
				print( "The input components were: ", amp, cen, width)
				print( "The output amplitudes, centroids and widths were: ", co_eff[0::3], co_eff[1::3], co_eff[2::3])
				print( "Test spectrum ID: ", ii)


	### print out the median errors and the interquartile range
	a_e = 100*numpy.array(a_e)
	c_e = 100*numpy.array(c_e)
	w_e = 100*numpy.array(w_e)

	print( " ")
	print( " ")
	print( " ")
	print( "##############################################")
	print( "#### Results for the single Gaussian test ####")
	print( "##############################################")

	print( "Median error on amplitude as a percentage = %.2f + %.2f - %.2f" %(numpy.percentile(a_e, 50), (numpy.percentile(a_e,75)-numpy.percentile(a_e,50)), (numpy.percentile(a_e,50)-numpy.percentile(a_e,25))))
	print( "Median error on centroids as a percentage = %.2f + %.2f - %.2f" %(numpy.percentile(c_e, 50), (numpy.percentile(c_e,75)-numpy.percentile(c_e,50)), (numpy.percentile(c_e,50)-numpy.percentile(c_e,25))))
	print( "Median error on widths as a percentage    = %.2f + %.2f - %.2f" %(numpy.percentile(w_e, 50), (numpy.percentile(w_e,75)-numpy.percentile(w_e,50)) , (numpy.percentile(w_e,50)-numpy.percentile(w_e,25))))
	print( "The number of spectra identified to have more than 1 component was: %i (%.2f%%)" %(n2, n2/num_test * 100))

	if(plot_tag==1):
		matplotlib.pyplot.figure(1)
		matplotlib.pyplot.xlabel("Input amplitude")
		matplotlib.pyplot.ylabel("Output amplitude")
		matplotlib.pyplot.figure(2)
		matplotlib.pyplot.xlabel("Input centroid")
		matplotlib.pyplot.ylabel("Output centroid")
		matplotlib.pyplot.figure(3)
		matplotlib.pyplot.xlabel("Input width")
		matplotlib.pyplot.ylabel("Output width")
		matplotlib.pyplot.show()


	return



### Here we now start the multiple component test. This is to test the code's ability to determine the number of components in test spectra.

def multi_gaussian_test(param):

	### Unpack the parameter array

	num_test = param["test_number"]
	
	spec_min = param["test_spec_min"]
	spec_max = param["test_spec_max"]
	spec_nv  = param["test_spec_num"]

	noise_level =  param["test_noise"]

	amp_min = param["test_amplitude_min"]
	amp_max = param["test_amplitude_max"]
	cen_min = param["test_vel_cen_min"]
	cen_max = param["test_vel_cen_max"]
	wid_min = param["test_width_min"]
	wid_max = param["test_width_max"]

	plot_tag = param["test_plot_tag"]

	var_noise = param["variable_noise"]
	if(var_noise!=0):
		print("variable_noise should be set to 0 for this test. Setting it to 0 now.")
		param["variable_noise"] = 0

	### Construct the velocity array

	v = numpy.linspace(spec_min,spec_max,spec_nv)

	### Initiize the seed for the random number generators
	numpy.random.seed(4)

	### arrays to store the input number of peaks and the output number of peaks
	ni = numpy.zeros(num_test)
	nf = numpy.zeros(num_test)

	count = 0

	print( "#######################")
	print( "#### Test progress ####")
	print( "#######################")
	print( " ")

	for ii in range(0,num_test):

		if((ii+1)%numpy.int(num_test/10.)==0):
			print( "The test is %.2f %% complete" %(100*(ii+1)/numpy.float(num_test)))

		### determine the number of peaks, between 1 and 4
		npeak = numpy.random.rand(1)*4 + 1
		npeak = int(npeak)
		ni[ii] = npeak

		### arrays for amplitudes, centroids and widths of the components
		amp = numpy.zeros(npeak)
		cen = numpy.zeros(npeak)
		width = numpy.zeros(npeak)

		nc = 0
		iteration = 0

		### This while loop is to ensure that there are no unresolved components in the test spectra.
		while(nc<npeak):

			if(iteration>50):
				ni[ii] = 0
				break

			iteration = iteration + 1

			pass_num = 0

			amp[nc] = numpy.random.rand(1)*(amp_max - amp_min) + amp_min
			cen[nc] = numpy.random.rand(1)*(cen_max - cen_min) + cen_min
			width[nc] = numpy.random.rand(1)*(wid_max - wid_min) + wid_min

			if(nc>0):

				### check between the new component and previous saved components to see if they are sufficiently far away to be resolved.
				for jj in range(0,nc):

					if(numpy.fabs(cen[nc] - cen[jj]) > 0.55*2.35*(width[nc] + width[jj])):

						pass_num = pass_num + 1

				### if the new component is resolved with respect to all the previous components then it is passed and may be saved
				if(pass_num == nc):
					nc = nc + 1

			else:
				nc = nc + 1

			
		if(iteration>50):
			nf[ii] = 0
			continue

		test_coeff = numpy.zeros(3*npeak)
		test_coeff[::3] = amp
		test_coeff[1::3] = cen
		test_coeff[2::3] = width 

		y = multi_gauss(v,test_coeff)
		mask = numpy.zeros_like(y)
		no = numpy.random.normal(loc=0.0,scale=noise_level,size=spec_nv)
		y = y+no
		mask[y>3*noise_level] = 1

		### Fit this synthetic spectrum
		co_eff,errors,AIC =fit_single_line(v,y,mask,param)

		### Check that a fit was found
		if(co_eff[0]==-1):
			nf[ii] = 0
			print( "No components detected")
			print( "The amplitude, centroid and width were: ", amp, cen, width)
			

		### If a fit was found then determine the number of components and find it.
		else:
			nf[ii] = len(co_eff[::3])

		### If the number of fitted components is not equal to the input number store this information.
		if(ni[ii]!=nf[ii]):
			count = count+1
			print("")
			print("The number of fitted components does not equal that of the input spectrum")
			print("Co-efficients for the input spectrum:")
			print(test_coeff)
			print("Co-efficients for the model spectrum:")
			print(co_eff)
			print("")

	print( " ")
	print( " ")
	print( " ")
	print( "################################################")
	print( "#### Results for the multiple Gaussian test ####")
	print( "################################################")

	print( "%d test spectra were fitted with the wrong number of components out of %d tests" %(count,len(ni[ni>0])))
	print( "Thus the code has a %.2f %% success rate for these parameters" %(100*(1-count/numpy.float(len(ni[ni>0])))))

	return

##### Function for reading parameter file

def read_parameters(param_file):


	### The dictionaries for the type of variable and the variable itself

	type_of_var = {"debug"                             :   "int",
                   "smoothing_length"                  :   "float",
	               "variable_noise"                    :   "int",
	               "noise_level"                       :   "float",
	               "signal_to_noise_ratio"             :   "float",
	               "max_peaks"                         :   "int",
	               "max_iterations"                    :   "int",
	               "min_velocity_channels"             :   "int",
	               "min_width_value"                   :   "float",
	               "max_width_value"                   :   "float",
	               "mask_pad"                          :   "int",
	               "delta_AIC_limit"                   :   "float",
	               "output_base"                       :   "str",
	               "velocity_channel_number"           :   "int",
	               "upper_sigma_level"                 :   "float",
	               "lower_sigma_level"                 :   "float",
	               "mask_filter_size"                  :   "int",
	               "use_velocity_range"                :   "int",
	               "min_velocity_range"                :   "float",
	               "max_velocity_range"                :   "float",
                   "test_number"                       :   "int",
                   "test_spec_min"                     :   "float",
                   "test_spec_max"                     :   "float",
                   "test_spec_num"                     :   "int",
                   "test_noise"                        :   "float",
                   "test_amplitude_min"                :   "float",
                   "test_amplitude_max"                :   "float",
                   "test_width_min"                    :   "float",
                   "test_width_max"                    :   "float",
                   "test_vel_cen_min"                  :   "float",
                   "test_vel_cen_max"                  :   "float",
                   "test_plot_tag"                     :   "int",
                   "data_in_file_name"                 :   "str"}

	param = {"debug"                             :   0,
             "smoothing_length"                  :   3.0,
	         "variable_noise"                    :   0,
	         "noise_level"                       :   0.1,
	         "signal_to_noise_ratio"             :   5,
	         "max_peaks"                         :   3,
	         "max_iterations"                    :   5,
	         "min_velocity_channels"             :   3,
	         "min_width_value"                   :   0.1,
	         "max_width_value"                   :   20.0,
	         "mask_pad"                          :   2,
	         "delta_AIC_limit"                   :   10.0,
	         "output_base"                       :   "output",
	         "velocity_channel_number"           :   30,
	         "upper_sigma_level"                 :   8,
	         "lower_sigma_level"                 :   4,
	         "mask_filter_size"                  :   3,
	         "use_velocity_range"                :   0,
	         "min_velocity_range"                :   -10,
	         "max_velocity_range"                :   10,
             "test_number"                       :   1000,
             "test_spec_min"                     :   -3.0,
             "test_spec_max"                     :   3.0,
             "test_spec_num"                     :   75,
             "test_noise"                        :   0.1,
             "test_amplitude_min"                :   1.0,
             "test_amplitude_max"                :   5.0,
             "test_width_min"                    :   0.5,
             "test_width_max"                    :   1.0,
             "test_vel_cen_min"                  :   -2.5,
             "test_vel_cen_max"                  :   2.5,
             "test_plot_tag"                     :   0,
             "data_in_file_name"                 :  "input.fits"}


	### Open the file and read through, ignoring comments.

	with open(param_file) as f:

		for line in f:

			if(line=="\n"):
				continue

			if(line[0]=="#"):
				continue

			words = line.split()
	

			try:

				var = type_of_var[words[0]]

				if(var=="str"):
					param[words[0]]=words[2]
				elif(var=="int"):
					param[words[0]]=numpy.int(words[2])                    
				elif(var=="float"):
					param[words[0]]=numpy.float(words[2])
				else:

					print( "The variable is neither a string, float or integer. I don't know how to deal with this")

			except KeyError:

				print( "There is no such parameter. Add it to the type_of_var and param dictionaries")


		f.close()

	### print the parameters to screen

	print( " ")
	print( " ")
	print( " ")
	print( "############################################")
	print( "################ Parameters ################")
	print( "############################################")
	print( "")
	print( "############# Important three ##############")
	print( "Delta AIC limit                 = ", param["delta_AIC_limit"])
	print( "Smoothing length                = ", param["smoothing_length"])
	print( "Signal to noise ratio           = ", param["signal_to_noise_ratio"])
	print( " ")

	print( "############# Input/Output names ###########")
	print( "Input Fits file name            = ", param["data_in_file_name"])
	print( "Output file base name           = ", param["output_base"])
	print( " ")

	print( "######## Spectral fitting parameters #######")
	print( "Maximum number of peaks allowed = ", param["max_peaks"])
	print( "Maximum number of iterations    = ", param["max_iterations"])
	print( "Minimum number of channels      = ", param["min_velocity_channels"])
	print( "Minimum component width         = ", param["min_width_value"])
	print( "Maximum component width         = ", param["max_width_value"])
	print( "Mask padding value              = ", param["mask_pad"])
	print( " ")

	print( "######## Non-test noise parameters #########")
	print( "Variable noise                  = ", param["variable_noise"])
	if(param["variable_noise"]==0):
		print( "Constant noise level            = ", param["noise_level"])
	print( " ")

	print( "######### Moment-masking parameters ########")
	print( "Noise velocity channel width    = ", param["velocity_channel_number"])
	print( "Upper sigma level for masking   = ", param["upper_sigma_level"])
	print( "Lower sigma level for masking   = ", param["lower_sigma_level"])
	print( "Top-hat filter size for masking = ", param["mask_filter_size"])
	print( "Limited velocity range flag     = ", param["use_velocity_range"])
	if(param["use_velocity_range"]==1):
		print( "Minimum velocity of the range   = ", param["min_velocity_range"])
		print( "Maximum velocity of the range   = ", param["max_velocity_range"])
	print( " ")

	print( "################## Flags ###################")
	print( "Debug switch                    = ", param["debug"])
	print( " ")

	print( "######### Parameters for test runs #########")
	print( "Number of test spectra          = ", param["test_number"])
	print( "Spectral range                  = [", param["test_spec_min"],",",param["test_spec_max"],"]")
	print( "Number of spectral bins         = ", param["test_spec_num"])
	print( "Test noise level                = ", param["test_noise"])
	print( "Amplitude range of components   = [", param["test_amplitude_min"],",",param["test_amplitude_max"],"]")
	print( "Velocity centroid range         = [", param["test_vel_cen_min"],",",param["test_vel_cen_max"],"]")
	print( "Width range of components       = [", param["test_width_min"],",",param["test_width_max"],"]")
	print( "Plotting switch                 = ", param["test_plot_tag"])
	print( " ")

	print( " ")
	print( " ")

	return param







##### Function which fits a whole fits cube if the fits cube has all the right information in the header

def fit_a_fits(param):

	### Read the parameter file and get the input fits file name

	fitsfile = param["data_in_file_name"]
	maskfile = param["output_base"]+"_mask.fits"
	max_peaks = param["max_peaks"]
	
	### open the fits file with astropy and extract the ppv datacube

	spec_fits = astropy.io.fits.open(fitsfile)
	mask_fits = astropy.io.fits.open(maskfile)
	ppv_data = spec_fits[0].data
	mask_data = mask_fits[0].data

	### check to make sure the datacube is 3D

	if(numpy.shape(ppv_data)[0] == 1):
		ppv_data = ppv_data[0,:,:,:]
	if(numpy.shape(mask_data)[0] ==1):
		mask_data = mask_data[0,:,:,:]

	### Extract the required information from the fits file header to construct the velocity axis

	neg_dv = 0
	vel = get_vel(spec_fits[0].header)
	dv = vel[1] - vel[0]
	nv = len(vel)

	if(dv<0):
		vel = numpy.flip(vel,axis=0)
		ppv_data = numpy.flip(ppv_data,axis=0)
		mask_data = numpy.flip(mask_data,axis=0)
		dv = numpy.fabs(dv)
		neg_dv = 1

	minv = vel[0]
	maxv = vel[-1]

	### Get the size of the ppv datacube and allocate the arrays for the output.

	nx = len(ppv_data[0,0,:])
	ny = len(ppv_data[0,:,0])

	amp_out = numpy.zeros((max_peaks,ny,nx))
	amp_err = numpy.zeros_like(amp_out)

	cen_out = numpy.zeros((max_peaks,ny,nx))
	cen_err = numpy.zeros_like(cen_out)

	wid_out = numpy.zeros((max_peaks,ny,nx))
	wid_err = numpy.zeros_like(wid_out)

	mod_out = numpy.zeros_like(ppv_data)
	res_out = numpy.zeros_like(ppv_data)

	### Just counters
	count = numpy.zeros(max_peaks)
	no_con = 0


	print( "##########################")
	print( "#### Fitting progress ####")
	print( "##########################")

	### Loop over the lines of sight in the datacube and try to fit each one

	for ii in range(0,ny):
		if((ii+1)%numpy.int(ny/10.)==0):
			print( "We are %.2f %% of the way through the cube" %(100*(ii+1)/numpy.float(ny)))
		for jj in range(0,nx):

			### Take the spectrum from the line of sight
			spec = ppv_data[:,ii,jj]
			mask = mask_data[:,ii,jj]

			### Make sure there are no nans in the spectrum
			nan_check = numpy.sum(numpy.isnan(spec))
			if(nan_check>0):
				continue

			### Fit spectrum and unpack the outputs
			co_eff,errors,AIC =fit_single_line(vel,spec,mask,param)
			mod_out[:,ii,jj] = multi_gauss(vel,co_eff)

			if(param["variable_noise"]==1):
				res_out[:,ii,jj] = (spec - mod_out[:,ii,jj])/numpy.std(spec[mask==0])
			if(param["variable_noise"]==0):
				res_out[:,ii,jj] = (spec - mod_out[:,ii,jj])/param["noise_level"]

			### Check for no convergence or if no peak was detected.			

			if(co_eff[0] == -1):
				no_con = no_con + 1
				continue
			if(co_eff[0] < 0):
				continue

			### If there was a fit, work out the number of components fitted and store this number in the count array	

			n = len(co_eff[::3])
			count[n-1] = count[n-1] + 1

			### Loop over the components found and store in the 3 output arrays

			for kk in range(0,n):

				amp_out[kk,ii,jj] = co_eff[3*kk]
				cen_out[kk,ii,jj] = co_eff[3*kk + 1]
				wid_out[kk,ii,jj] = co_eff[3*kk + 2]

				amp_err[kk,ii,jj] = errors[3*kk]
				cen_err[kk,ii,jj] = errors[3*kk + 1]
				wid_err[kk,ii,jj] = errors[3*kk + 2]





	### print out results

	print( " ")
	print( " ")
	print( " ")
	print( "#########################")
	print( "###### Fit results ######")
	print( "#########################")

	for ii in range(0,max_peaks):
		if(count[ii]==0):
			continue
		else:
			print( "There are %d spectra with %d component, %.2f %% of total fitted spectra" %(count[ii], ii+1, 100*numpy.float(count[ii])/numpy.sum(count)))

	print( " ")
	if(no_con>0):
		print( "There was no convergence for %d spectra" %no_con)


	### Set up output. Copy the header from the input file and use that for the output files

	output_base = param["output_base"]
	hdu = astropy.io.fits.PrimaryHDU()
	
	### If there dv was negative we need to flip the model and residual cubes back around before outputting them
	if(neg_dv==1):
		mod_out = numpy.flip(mod_out,axis=0)
		res_out = numpy.flip(res_out,axis=0)

	### Collect all the co-efficients and their errors into arrays
	coeff_out = numpy.array([amp_out,cen_out,wid_out])
	error_out = numpy.array([amp_err,cen_err,wid_err])

	### Output co-efficients

	hdu.data = coeff_out
	outfile = "Coeff_"+output_base+".fits"
	hdu.writeto(outfile,overwrite=True)

	### Output errors

	hdu.data = error_out
	outfile = "Error_"+output_base+".fits"
	hdu.writeto(outfile,overwrite=True)

	### Output Model

	hdu.data = mod_out
	hdu.header = spec_fits[0].header
	outfile = "Model_"+output_base+".fits"
	hdu.writeto(outfile,overwrite=True)

	### Output residuals

	hdu.data = res_out
	hdu.header = spec_fits[0].header
	outfile = "Res_"+output_base+".fits"
	hdu.writeto(outfile,overwrite=True)

	return


### Function to make a 3D top-hat function and convolve it with the datacube to make a smoothed cube
def TopHat_3DFilter(Image, Filter_size):

	kernel = numpy.ones((Filter_size,Filter_size,Filter_size))
	kernel = kernel / numpy.sum(kernel)
	Final_image = convolve_fft(Image,kernel,boundary="wrap")

	return Final_image



### Take the smoothed cube and determine the mask needed for moment-masking
def make_mask(image,data,param):

	### unpack needed parameters and open fits file
	fitsfile = param["data_in_file_name"]
	maskfile = param["output_base"]+"_mask.fits"
	nvel = param["velocity_channel_number"]
	upper = param["upper_sigma_level"]
	lower = param["lower_sigma_level"]
	spec_fits = astropy.io.fits.open(fitsfile)

	### Determine the noise level on a pixel-by-pixel basis using the number of velocity channels (nvel) specified in the parameter file
	noise = numpy.zeros_like(image)
	image[numpy.isnan(image)] = 0

	for ii in range(0,image.shape[1]):
		for jj in range(0,image.shape[2]):

			if(nvel>0):
				noise[:,ii,jj] = numpy.nanstd(image[:nvel,ii,jj])
			if(nvel<0):
				noise[:,ii,jj] = numpy.nanstd(image[nvel:,ii,jj])


	# Set up the mask 3D array and set everything above the upper limit to 1
	mask = numpy.zeros_like(image)
	mask[image>upper*noise] = 1

	# Initialise counters etc.
	change = 1
	tot_num = 0
	old_tot_num = numpy.sum(mask)
	counter = 1

	print( "##########################")
	print( "#### Masking progress ####")
	print( "##########################")


	### While the number of voxels in the mask keeps changing keep iterating
	while(change==1):

		### Find all voxels which are above the lower limit and are not currently masked
		index = numpy.where((mask==0)*(image > lower*noise))

		ix = index[0]
		iy = index[1]
		iz = index[2]

		nx = len(ix)

		### Loop over these candidate voxels and check if one of their 6 neighbours is currently masked 
		for aa in range(0,nx):

			ii = ix[aa]
			jj = iy[aa]
			kk = iz[aa]

			if(ii==0 or jj==0 or kk==0 or ii==mask.shape[0]-1 or jj==mask.shape[1]-1 or kk==mask.shape[2]-1 ):
				continue

			neighbour = 0
			neighbour = neighbour + mask[ii-1,jj,kk]
			neighbour = neighbour + mask[ii+1,jj,kk]

			neighbour = neighbour + mask[ii,jj-1,kk]
			neighbour = neighbour + mask[ii,jj+1,kk]

			neighbour = neighbour + mask[ii,jj,kk-1]
			neighbour = neighbour + mask[ii,jj,kk+1]

			## If a neighbour is masked then mask this voxel
			if(neighbour>0):

				mask[ii,jj,kk]=1

		tot_num = numpy.sum(mask)

		if(tot_num == old_tot_num):
			change = 0
		else:
			print("Iteration ", counter, " the old number of mask pixels was ", old_tot_num, ", now it is ", tot_num)
			old_tot_num = tot_num
			counter=counter+1

	print("")
	print("")
	print("")

	### Ensure that the mask is 0 if the unsmoothed data is NaN or below zero for this voxel
	mask[numpy.isnan(data)] = 0
	mask[data<0] = 0

	### Output the mask to a fits file
	hdu = astropy.io.fits.PrimaryHDU()
	hdu.header = spec_fits[0].header
	hdu.header["BSCALE"]  =  1                                        
	hdu.header["BZERO"]   =  0
	hdu.data = mask
	hdu.writeto(maskfile,overwrite=True)

	return mask

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    av = numpy.average(values, weights=weights)
    # Fast and numerically precise:
    variance = numpy.average((values-av)**2, weights=weights)
    return (av, numpy.sqrt(variance))


### A function to calculate the moment internally
def make_moments_int(cube,mask,param):

	#Unpack parameters and open header
	moms_out = param["output_base"]
	fitsfile = param["data_in_file_name"]
	vel_range_flag = param["use_velocity_range"]
	data_head = astropy.io.fits.getheader(fitsfile)
	
	#Construct velocity array from fits file header and check for negative dv
	vel = get_vel(data_head)
	dv = vel[1] - vel[0]
	if(dv<0):
		dv = dv*-1

	if(vel_range_flag==1):
		vel_min_range = param["min_velocity_range"]
		vel_max_range = param["max_velocity_range"]

		vv = numpy.array([vel]*cube.shape[1])
		vv = numpy.array([vv]*cube.shape[2])
		vv = vv.T

		mask[vv<vel_min_range] = 0
		mask[vv>vel_max_range] = 0


	#Calculate moment zero from the sum along the velocity axis
	mom0 = numpy.sum(mask*cube , axis=0) * dv

	#Loop over all the pixel and determine the moment 1 and 2, as well as noise
	mom1 = numpy.zeros_like(mom0)
	mom2 = numpy.zeros_like(mom0)
	noise = numpy.zeros_like(mom0)
	for ii in range(0,mom1.shape[0]):
		for jj in range(0,mom1.shape[1]):

			## Determine noise first using only noisy channels, i.e. mask==0
			q = (mask[:,ii,jj]==0)
			noise_spec = cube[q,ii,jj]
			noise[ii,jj] = numpy.nanstd(noise_spec)

			## Determine moments from only emission channels, i.e. mask==1
			q = (mask[:,ii,jj]==1)
			v = vel[q]
			c = cube[q,ii,jj]
			# Ensure that there are at least 3 velocity channels with emission so that an actual moment 2 can be determined.
			if(numpy.sum(q)<3 or numpy.sum(c)<=0):
				continue
			mom1[ii,jj], mom2[ii,jj] = weighted_avg_and_std(v,c)


	### Set up the header for the moment maps
	hdu = astropy.io.fits.PrimaryHDU()
	hdu.header["NAXIS"] = 2

	keywords = ["NAXIS1","NAXIS2","CTYPE1","CRVAL1","CDELT1","CRPIX1","CUNIT1","CTYPE2","CRVAL2","CDELT2","CRPIX2","CUNIT2","RADESYS","EQUINOX","EPOCH"]

	for ii in range(0,len(keywords)):
		try:
			hdu.header[keywords[ii]] = data_head[keywords[ii]]
		except KeyError:
			print("No %s keyword in the header" %keywords[ii])

	print(" ")

	### Save the moment maps
	hdu.data = mom0
	hdu.writeto(moms_out+"_mom0.fits",overwrite=True)

	hdu.data = mom1
	hdu.writeto(moms_out+"_mom1.fits",overwrite=True)

	hdu.data = mom2
	hdu.writeto(moms_out+"_mom2.fits",overwrite=True)

	hdu.data = noise
	hdu.writeto(moms_out+"_noise.fits",overwrite=True)

	return 


### Constructs the velocity array assuming that velocity is the third axis in the fits cube
def get_vel(head):

	### If the header data is stored as frequency then convert to velocity [in km/s]
	if(head["CTYPE3"][0] == "F"):

		df = head['CDELT3']
		nf = head["CRPIX3"] 
		fr = head["CRVAL3"]

		ff = numpy.zeros(head["NAXIS3"])
		for ii in range(0,len(ff)):
			ff[ii] = fr + (ii-nf+1)*df

		rest = head["RESTFRQ"]

		vel = (rest-ff) / rest * 299792.458 
		return vel

	elif(head["CTYPE3"][0] == "V"):

		refnv = head["CRPIX3"]
		refv = head["CRVAL3"]
		dv = head["CDELT3"]
		### Construct the velocity axis 

		vel = numpy.zeros(head["NAXIS3"])
		for ii in range(0,len(vel)):
			vel[ii] = refv + (ii-refnv+1)*dv

		return vel

	else:

		print("The CTYPE3 variable in the fitsfile header does not start with F for frequency or V for velocity")
		return


### Function to be called externally to calculate the moments
def make_moments(param):

	## Unpack parameters and open data file
	fitsfile = param["data_in_file_name"]
	Filter_size = param["mask_filter_size"]
	data_head = astropy.io.fits.getheader(fitsfile)
	data = astropy.io.fits.getdata(fitsfile)
	
	# If the data has a 4 dimension, turn it into 3D
	if(numpy.shape(data)[0] == 1):
		data = data[0,:,:,:]

	# Smooth the data, make a mask, and then make the moments
	data_copy = numpy.copy(data)
	data_copy[numpy.isnan(data_copy)] = 0
	data_s = TopHat_3DFilter(data_copy,Filter_size)
	mask = make_mask(data_s,data,param)
	make_moments_int(data,mask,param)



