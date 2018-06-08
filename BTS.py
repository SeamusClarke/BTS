import numpy
import matplotlib.pyplot
from scipy.optimize import curve_fit
from astropy.convolution import convolve, Gaussian1DKernel
import astropy.io.fits
import os

############ Fitting routine itself

def fit_single_line(vel,x,params):

	####### Unpack the parameter array

	chi_limit = params["chi_limit"]
	overlap_tag = params["check_overlap"]
	debug = params["debug"]
	lowerlimit = params["lower_integrated_emission_limit"]
	smooth = params["smoothing_length"]
	var_noise = params["variable_noise"]
	noise_level = params["noise_level"]
	noise_clip = params["noise_clip"]
	n = params["signal_to_noise_ratio"]

	if(debug==1):
		print "##########"
		print "DEBUG MODE"
		print "##########"
		print " "

	overlap=0

	nv = len(x)
	maxv=max(vel)
	minv=min(vel)
	dv = vel[1] - vel[0]

	####### Determine the noise level

	if(var_noise == 1):
		noise = numpy.std(x[:noise_clip])
	else:
		noise = noise_level
		
	#### prepare the data and convolve spectrum with gaussian for peak determining.

	spec = x[:]
	gk = Gaussian1DKernel(smooth)
			
	spec3 = convolve(spec,gk)

	### if the integrated intensity of the line of sight is smaller than a given limit we skip

	if(sum(spec)*dv < lowerlimit):
		return [[-1,0,0],[0]]

	### Work out the gradients of the spectrum

	dspec = numpy.zeros_like(spec)
	for ii in range(0,nv-1):
		dspec[ii] = (spec3[ii+1]-spec3[ii])/dv

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
			if(ddspec[ii+1] > ddspec[ii] and ddspec[ii+2] > ddspec[ii] and dddspec[ii]>0.0):
				if(decrease==1 and (spec[ii]>n*noise or spec[ii+1] > n*noise or spec[ii-1] > n*noise )):	
					switch[ii] = 1
				decrease = 0

	index = numpy.linspace(0,nv-1,nv)
	index = numpy.array(index,dtype=numpy.int)


	### if there are no peaks then skip pixel

	if(sum(switch)<1):
		return [[-1,0,0],[0]]

	### here we set up the arrays that contain the guesses for the peaks' amplitudes and centriods and widths

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

			ran = vel[finish] -vel[start]

			l = len(pid[(pid<finish)*(pid>start)])

			for jj in range(num,num+l):
				psig[jj] = ran/(l*numpy.sqrt(8*numpy.log(2)))

			num=num+l
			
			if(debug==1):
				print "start  = ", vel[start]
				print "finish = ", vel[finish]

			start=-1
			finish=-1


	### if a guess width is smaller than the velocity resolution then we set it to the velocity resoution

	psig[psig<dv] = dv

	n_peaks = len(pamp)

	if(debug==1):
	
		print "######## Guess values #########"
		print "Number of peaks = ", n_peaks
		print "Peak ids = ", pid
		print "Peak centroids = ", pcent
		print "Peak amplitude = ", pamp
		print "Peak width = ", psig
		print " "

	### if more than 6 components were detected then lets try fitting with 6

	if(n_peaks>6):

		print "More than 6 peaks were detected. We will try to fit with just 6"
		pamp=pamp[:6]
		psig=psig[:6]
		pcent = pcent[:6]
		pid = pid[:6]
		n_peaks = 6
	
		if(debug==1):
			matplotlib.pyplot.plot(vel,spec)
			matplotlib.pyplot.show()
			print pamp
			print psig
			print pcent

	guess = numpy.zeros(3*n_peaks)
	bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
	guess = numpy.array(guess,dtype=numpy.double)


	### set limits on the guess and fill the guess and boundary arrays
	
	for ii in range(0,n_peaks):

		if(pamp[ii]<n*noise):
			pamp[ii]=n*noise+0.01
		if(pcent[ii]>maxv):
			pcent[ii]=maxv-dv
		if(pcent[ii]<minv):
			pcent[ii]=minv+dv
		if(psig[ii]>maxv-minv):
			psig[ii] = 0.99*(maxv-minv)
		if(psig[ii]<dv):
			psig[ii]=dv+0.01

		guess[3*ii] = pamp[ii]
		guess[3*ii+1] = pcent[ii]
		guess[3*ii+2] = psig[ii]

		bound[0][3*ii] = n*noise
		bound[1][3*ii] = 2*max(spec)
		bound[0][3*ii+1] = minv
		bound[1][3*ii+1] = maxv
		bound[0][3*ii+2] = dv
		bound[1][3*ii+2] = maxv-minv


	### Fill the arrays for the guesses which contain fewer peaks than detected. This is a safety feature against over-fitting

	if(n_peaks>1):
		bound3 = (numpy.zeros(3*n_peaks -3),numpy.zeros(3*n_peaks - 3))

		for ii in range(0,n_peaks-1):

			bound3[0][3*ii] = n*noise
			bound3[1][3*ii] = 2*max(spec)
			bound3[0][3*ii+1] = minv
			bound3[1][3*ii+1] = maxv
			bound3[0][3*ii+2] = dv
			bound3[1][3*ii+2] = maxv-minv

		guess3 = numpy.zeros(3*n_peaks-3)
		min_amp_index = numpy.argmin(guess[::3])
		c_num=0
		for ii in range(0,n_peaks):

			if(ii!=min_amp_index):
				
				guess3[3*c_num:3*c_num+3:1] = guess[(3*ii):(3*ii)+3:1]
				c_num=c_num+1


	if(n_peaks>2):
		bound4 = (numpy.zeros(3*n_peaks -6),numpy.zeros(3*n_peaks - 6))

		for ii in range(0,n_peaks-2):

			bound4[0][3*ii] = n*noise
			bound4[1][3*ii] = 2*max(spec)
			bound4[0][3*ii+1] = minv
			bound4[1][3*ii+1] = maxv
			bound4[0][3*ii+2] = dv
			bound4[1][3*ii+2] = maxv-minv

		guess4 = numpy.zeros(3*n_peaks-6)
		min_amp_index = numpy.argmin(guess3[::3])
		c_num=0
		for ii in range(0,n_peaks-1):

			if(ii!=min_amp_index):
				
				guess4[3*c_num:3*c_num+3] = guess3[3*ii:3*ii+3]
				c_num=c_num+1


	if(n_peaks>3):
		bound5 = (numpy.zeros(3*n_peaks -9),numpy.zeros(3*n_peaks - 9))

		for ii in range(0,n_peaks-3):

			bound5[0][3*ii] = n*noise
			bound5[1][3*ii] = 2*max(spec)
			bound5[0][3*ii+1] = minv
			bound5[1][3*ii+1] = maxv
			bound5[0][3*ii+2] = dv
			bound5[1][3*ii+2] = maxv-minv

		guess5 = numpy.zeros(3*n_peaks-9)
		min_amp_index = numpy.argmin(guess4[::3])
		c_num=0
		for ii in range(0,n_peaks-2):

			if(ii!=min_amp_index):
				
				guess5[3*c_num:3*c_num+3] = guess4[3*ii:3*ii+3]
				c_num=c_num+1


	if(n_peaks>4):
		bound6 = (numpy.zeros(3*n_peaks -12),numpy.zeros(3*n_peaks - 12))

		for ii in range(0,n_peaks-4):

			bound6[0][3*ii] = n*noise
			bound6[1][3*ii] = 2*max(spec)
			bound6[0][3*ii+1] = minv
			bound6[1][3*ii+1] = maxv
			bound6[0][3*ii+2] = dv
			bound6[1][3*ii+2] = maxv-minv

		guess6 = numpy.zeros(3*n_peaks-12)
		min_amp_index = numpy.argmin(guess5[::3])
		c_num=0
		for ii in range(0,n_peaks-3):

			if(ii!=min_amp_index):
				
				guess6[3*c_num:3*c_num+3] = guess5[3*ii:3*ii+3]
				c_num=c_num+1


	if(n_peaks>5):
		bound7 = (numpy.zeros(3*n_peaks -15),numpy.zeros(3*n_peaks - 15))

		for ii in range(0,n_peaks-5):

			bound7[0][3*ii] = n*noise
			bound7[1][3*ii] = 2*max(spec)
			bound7[0][3*ii+1] = minv
			bound7[1][3*ii+1] = maxv
			bound7[0][3*ii+2] = dv
			bound7[1][3*ii+2] = maxv-minv

		guess7 = numpy.zeros(3*n_peaks-15)
		min_amp_index = numpy.argmin(guess6[::3])
		c_num=0
		for ii in range(0,n_peaks-4):

			if(ii!=min_amp_index):
				
				guess7[3*c_num:3*c_num+3] = guess6[3*ii:3*ii+3]
				c_num=c_num+1




	### Here we fit a single peak

	
	if(n_peaks==1):
		r_chi_sq, res, co_eff, var_matrix, converged = fit1(vel,spec,guess,bound,noise)

		
		if(debug==1):
			print "########## First fit ###########"
			print "Number of peaks = ", n_peaks
			print "Reduced chi_sq = ", r_chi_sq
			print "Residuals = ", res
			print "Co-effs = ", co_eff
			print "Converged = ", converged
			print " "
			

	### If two peaks were fitted we fit with 2 and also 1 peak. Then we compare the reduced chi_sq to the chi_sq limit from the parameters. If the 1 peak fit is sufficiently good, we keep it.

	elif(n_peaks==2):
		r_chi_sq, res, co_eff, var_matrix, converged = fit2(vel,spec,guess,bound,noise)
		
		r_chi_sq2, res2, co_eff2, var_matrix2, converged2 = fit1(vel,spec,guess3,bound3,noise)

		if(debug==1):
			print "########## First fit ###########"
			print "Number of peaks = ", n_peaks
			print " "
			print "For the 2 peak fit"
			print "Reduced chi_sq = ", r_chi_sq
			print "Residuals = ", res
			print "Co-effs = ", co_eff
			print "Converged = ", converged
			print " "
			print "For the 1 peak fit"
			print "Reduced chi_sq = ", r_chi_sq2
			print "Residuals = ", res2
			print "Co-effs = ", co_eff2
			print "Converged = ", converged2
			print " "

		if(r_chi_sq2<chi_limit and converged2==1):

			n_peaks=1

			guess = numpy.zeros(3*n_peaks)
			bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
			guess = numpy.array(guess,dtype=numpy.double)

			co_eff = numpy.zeros(3*n_peaks)
			var_matrix = numpy.zeros((3*n_peaks,3*n_peaks))

			guess = guess3
			bound = bound3
			r_chi_sq = r_chi_sq2
			res = res2
			co_eff = co_eff2
			var_matrix = var_matrix2
			converged = converged2

		if(debug==1):
			print "####### After check ########"
			print "Number of peaks = ", n_peaks
			print " "


	### If 3 peaks detected we try fitting with 3, 2 and 1 peaks. 
		
	elif(n_peaks==3):
		r_chi_sq, res, co_eff, var_matrix, converged = fit3(vel,spec,guess,bound,noise)

		r_chi_sq2, res2, co_eff2, var_matrix2, converged2 = fit2(vel,spec,guess3,bound3,noise)

		r_chi_sq3, res3, co_eff3, var_matrix3, converged3 = fit1(vel,spec,guess4,bound4,noise)


		if(debug==1):
			print "########## First fit ###########"
			print "Number of peaks = ", n_peaks
			print " "
			print "For the 3 peak fit"
			print "Reduced chi_sq = ", r_chi_sq
			print "Residuals = ", res
			print "Co-effs = ", co_eff
			print "Converged = ", converged
			print " "
			print "For the 2 peak fit"
			print "Reduced chi_sq = ", r_chi_sq2
			print "Residuals = ", res2
			print "Co-effs = ", co_eff2
			print "Converged = ", converged2
			print " "
			print "For the 1 peak fit"
			print "Reduced chi_sq = ", r_chi_sq3
			print "Residuals = ", res3
			print "Co-effs = ", co_eff3
			print "Converged = ", converged3
			print " "

		if(r_chi_sq3<chi_limit and converged3==1):

			n_peaks=1 

			guess = numpy.zeros(3*n_peaks)
			bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
			guess = numpy.array(guess,dtype=numpy.double)

			co_eff = numpy.zeros(3*n_peaks)
			var_matrix = numpy.zeros((3*n_peaks,3*n_peaks))

			guess = guess4
			bound = bound4
			r_chi_sq = r_chi_sq3
			res = res3
			co_eff = co_eff3
			var_matrix = var_matrix3
			converged = converged3

		elif(r_chi_sq2<chi_limit and converged2==1):

			n_peaks=2

			guess = numpy.zeros(3*n_peaks)
			bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
			guess = numpy.array(guess,dtype=numpy.double)

			co_eff = numpy.zeros(3*n_peaks)
			var_matrix = numpy.zeros((3*n_peaks,3*n_peaks))

			guess = guess3
			bound = bound3
			r_chi_sq = r_chi_sq2
			res = res2
			co_eff = co_eff2
			var_matrix = var_matrix2
			converged = converged2


	### If 4 peaks detected we try fitting 4, 3, 2 and 1 peaks

	elif(n_peaks==4):
		r_chi_sq, res, co_eff, var_matrix, converged = fit4(vel,spec,guess,bound,noise)

		r_chi_sq2, res2, co_eff2, var_matrix2, converged2 = fit3(vel,spec,guess3,bound3,noise)

		r_chi_sq3, res3, co_eff3, var_matrix3, converged3 = fit2(vel,spec,guess4,bound4,noise)

		r_chi_sq4, res4, co_eff4, var_matrix4, converged4 = fit1(vel,spec,guess5,bound5,noise)

		if(r_chi_sq4<chi_limit and converged4==1):

			n_peaks=1

			guess = numpy.zeros(3*n_peaks)
			bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
			guess = numpy.array(guess,dtype=numpy.double)

			co_eff = numpy.zeros(3*n_peaks)
			var_matrix = numpy.zeros((3*n_peaks,3*n_peaks))

			guess = guess5
			bound = bound5
			r_chi_sq = r_chi_sq4
			res = res4
			co_eff = co_eff4
			var_matrix = var_matrix4
			converged = converged4

		elif(r_chi_sq3<chi_limit and converged3==1):

			n_peaks=2 

			guess = numpy.zeros(3*n_peaks)
			bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
			guess = numpy.array(guess,dtype=numpy.double)

			co_eff = numpy.zeros(3*n_peaks)
			var_matrix = numpy.zeros((3*n_peaks,3*n_peaks))

			guess = guess4
			bound = bound4
			r_chi_sq = r_chi_sq3
			res = res3
			co_eff = co_eff3
			var_matrix = var_matrix3
			converged = converged2

		elif(r_chi_sq2<chi_limit and converged2==1):

			n_peaks=3

			guess = numpy.zeros(3*n_peaks)
			bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
			guess = numpy.array(guess,dtype=numpy.double)

			co_eff = numpy.zeros(3*n_peaks)
			var_matrix = numpy.zeros((3*n_peaks,3*n_peaks))

			guess = guess3
			bound = bound3
			r_chi_sq = r_chi_sq2
			res = res2
			co_eff = co_eff2
			var_matrix = var_matrix2
			converged = converged2



	### If 5 peaks detected, then fit with 5, 4, 3, 2 and 1.

	elif(n_peaks==5):
		r_chi_sq, res, co_eff, var_matrix, converged = fit5(vel,spec,guess,bound,noise)

		r_chi_sq2, res2, co_eff2, var_matrix2, converged2 = fit4(vel,spec,guess3,bound3,noise)

		r_chi_sq3, res3, co_eff3, var_matrix3, converged3 = fit3(vel,spec,guess4,bound4,noise)

		r_chi_sq4, res4, co_eff4, var_matrix4, converged4 = fit2(vel,spec,guess5,bound5,noise)

		r_chi_sq5, res5, co_eff5, var_matrix5, converged5 = fit1(vel,spec,guess6,bound6,noise)

		if(r_chi_sq5<chi_limit and converged5==1):

			n_peaks=1

			guess = numpy.zeros(3*n_peaks)
			bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
			guess = numpy.array(guess,dtype=numpy.double)

			co_eff = numpy.zeros(3*n_peaks)
			var_matrix = numpy.zeros((3*n_peaks,3*n_peaks))

			guess = guess6
			bound = bound6
			r_chi_sq = r_chi_sq5
			res = res5
			co_eff = co_eff5
			var_matrix = var_matrix5
			converged = converged5

		elif(r_chi_sq4<chi_limit and converged4==1):

			n_peaks=2

			guess = numpy.zeros(3*n_peaks)
			bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
			guess = numpy.array(guess,dtype=numpy.double)

			co_eff = numpy.zeros(3*n_peaks)
			var_matrix = numpy.zeros((3*n_peaks,3*n_peaks))

			guess = guess5
			bound = bound5
			r_chi_sq = r_chi_sq4
			res = res4
			co_eff = co_eff4
			var_matrix = var_matrix4
			converged = converged4

		elif(r_chi_sq3<chi_limit and converged3==1):

			n_peaks=3

			guess = numpy.zeros(3*n_peaks)
			bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
			guess = numpy.array(guess,dtype=numpy.double)

			co_eff = numpy.zeros(3*n_peaks)
			var_matrix = numpy.zeros((3*n_peaks,3*n_peaks))

			guess = guess4
			bound = bound4
			r_chi_sq = r_chi_sq3
			res = res3
			co_eff = co_eff3
			var_matrix = var_matrix3
			converged = converged3

		elif(r_chi_sq2<chi_limit and converged2==1):

			n_peaks=4

			guess = numpy.zeros(3*n_peaks)
			bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
			guess = numpy.array(guess,dtype=numpy.double)

			co_eff = numpy.zeros(3*n_peaks)
			var_matrix = numpy.zeros((3*n_peaks,3*n_peaks))

			guess = guess3
			bound = bound3
			r_chi_sq = r_chi_sq2
			res = res2
			co_eff = co_eff2
			var_matrix = var_matrix2
			converged = converged2


	### If 6 peaks detected, fit 6, 5, 4, 3, 2 and 1 peaks

	elif(n_peaks==6):
		r_chi_sq, res, co_eff, var_matrix, converged = fit6(vel,spec,guess,bound,noise)

		r_chi_sq2, res2, co_eff2, var_matrix2, converged2 = fit5(vel,spec,guess3,bound3,noise)

		r_chi_sq3, res3, co_eff3, var_matrix3, converged3 = fit4(vel,spec,guess4,bound4,noise)

		r_chi_sq4, res4, co_eff4, var_matrix4, converged4 = fit3(vel,spec,guess5,bound5,noise)

		r_chi_sq5, res5, co_eff5, var_matrix5, converged5 = fit2(vel,spec,guess6,bound6,noise)

		r_chi_sq6, res6, co_eff6, var_matrix6, converged6 = fit1(vel,spec,guess7,bound7,noise)

		if(r_chi_sq6<chi_limit and converged6==1):

			n_peaks=1

			guess = numpy.zeros(3*n_peaks)
			bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
			guess = numpy.array(guess,dtype=numpy.double)

			co_eff = numpy.zeros(3*n_peaks)
			var_matrix = numpy.zeros((3*n_peaks,3*n_peaks))

			guess = guess7
			bound = bound7
			r_chi_sq = r_chi_sq6
			res = res6
			co_eff = co_eff6
			var_matrix = var_matrix6
			converged = converged6

		elif(r_chi_sq5<chi_limit and converged5==1):

			n_peaks=2

			guess = numpy.zeros(3*n_peaks)
			bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
			guess = numpy.array(guess,dtype=numpy.double)

			co_eff = numpy.zeros(3*n_peaks)
			var_matrix = numpy.zeros((3*n_peaks,3*n_peaks))

			guess = guess6
			bound = bound6
			r_chi_sq = r_chi_sq5
			res = res5
			co_eff = co_eff5
			var_matrix = var_matrix5
			converged = converged5

		elif(r_chi_sq4<chi_limit and converged4==1):

			n_peaks=3

			guess = numpy.zeros(3*n_peaks)
			bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
			guess = numpy.array(guess,dtype=numpy.double)

			co_eff = numpy.zeros(3*n_peaks)
			var_matrix = numpy.zeros((3*n_peaks,3*n_peaks))

			guess = guess5
			bound = bound5
			r_chi_sq = r_chi_sq4
			res = res4
			co_eff = co_eff4
			var_matrix = var_matrix4
			converged = converged4

		elif(r_chi_sq3<chi_limit and converged3==1):

			n_peaks=4

			guess = numpy.zeros(3*n_peaks)
			bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
			guess = numpy.array(guess,dtype=numpy.double)

			co_eff = numpy.zeros(3*n_peaks)
			var_matrix = numpy.zeros((3*n_peaks,3*n_peaks))

			guess = guess4
			bound = bound4
			r_chi_sq = r_chi_sq3
			res = res3
			co_eff = co_eff3
			var_matrix = var_matrix3
			converged = converged3

		elif(r_chi_sq2<chi_limit and converged2==1):

			n_peaks=5

			guess = numpy.zeros(3*n_peaks)
			bound = (numpy.zeros(3*n_peaks),numpy.zeros(3*n_peaks))
			guess = numpy.array(guess,dtype=numpy.double)

			co_eff = numpy.zeros(3*n_peaks)
			var_matrix = numpy.zeros((3*n_peaks,3*n_peaks))

			guess = guess3
			bound = bound3
			r_chi_sq = r_chi_sq2
			res = res2
			co_eff = co_eff2
			var_matrix = var_matrix2
			converged = converged2




	### If no fit converged then we exit
	if(converged==0):

		print "No convergence"
		return [[0,0,0],[0]]

	### check for an overlap if this option is switched on 
	if(n_peaks>1 and overlap_tag == 1):
		overlap = check_overlap(co_eff,dv)

		if(debug==1):
			print "##### Check overlap #####"
			print "Overlap = ", overlap
			print " "
		

	### if we have an overlap then we need to drop a peak and fit with one fewer
	if(overlap==1 and n_peaks>1):

		bound2 = (numpy.zeros(3*n_peaks -3),numpy.zeros(3*n_peaks - 3))

		for ii in range(0,n_peaks-1):

			bound2[0][3*ii] = n*noise
			bound2[1][3*ii] = 2*max(spec)
			bound2[0][3*ii+1] = minv
			bound2[1][3*ii+1] = maxv
			bound2[0][3*ii+2] = dv
			bound2[1][3*ii+2] = maxv-minv

		tbr_index = numpy.argmax(guess[::3])
		guess2 = numpy.zeros(3*(n_peaks-1))
		guess2[:3*tbr_index] = guess[:3*tbr_index]
		if(3*tbr_index < 3*n_peaks - 3):
			guess2[3*tbr_index:] = guess[3*tbr_index + 3:]


		if(n_peaks==2):
			r_chi_sq2, res2, co_eff2, var_matrix2, converged = fit1(vel,spec,guess2,bound2,noise)
		if(n_peaks==3):
			r_chi_sq2, res2, co_eff2, var_matrix2, converged = fit2(vel,spec,guess2,bound2,noise)
		if(n_peaks==4):
			r_chi_sq2, res2, co_eff2, var_matrix2, converged = fit3(vel,spec,guess2,bound2,noise)
		if(n_peaks==5):
			r_chi_sq2, res2, co_eff2, var_matrix2, converged = fit4(vel,spec,guess2,bound2,noise)
		if(n_peaks==6):
			r_chi_sq2, res2, co_eff2, var_matrix2, converged = fit5(vel,spec,guess2,bound2,noise)

		if(debug==1):
			print "##### After overlap reduction #####"
			print "Number of peaks = ", n_peaks - 1
			print "Reduced chi_sq = ", r_chi_sq2
			print "Residuals = ", res2
			print "Co-effs = ", co_eff2
			print "Converged = ", converged
			print " "

		#### check convergence, if no convergence we do not output the old fit as it still contains overlapping components.
		if(converged==0):

			print "No convergence"
			return [[0,0,0],[0]]

		### if converged return the new fit
		return co_eff2,r_chi_sq2				


	
	## No overlap and the fit converged but it isn't that good
	if(overlap==0 and r_chi_sq > chi_limit and n_peaks<6):

		### Add more guesses and bounds. We take the location of maximum residuial as the place to add a component.
		guess2 = numpy.zeros(len(guess) + 3)
		guess2[:3*n_peaks] = guess[:3*n_peaks]

		high_res = numpy.argmax(numpy.fabs(res))

		guess2[3*n_peaks] = 1.01*n*noise
		guess2[3*n_peaks+1] = vel[high_res]
		guess2[3*n_peaks+2] = 1.01*dv


		bound2 = (numpy.zeros(3*n_peaks + 3),numpy.zeros(3*n_peaks + 3))

		for ii in range(0,n_peaks+1):

			bound2[0][3*ii] = n*noise
			bound2[1][3*ii] = 2*max(spec)
			bound2[0][3*ii+1] = minv
			bound2[1][3*ii+1] = maxv
			bound2[0][3*ii+2] = dv
			bound2[1][3*ii+2] = maxv-minv

		### Try fitting with this additional peak

		if(n_peaks==1):
			r_chi_sq2, res2, co_eff2, var_matrix2, converged = fit2(vel,spec,guess2,bound2,noise)
		if(n_peaks==2):
			r_chi_sq2, res2, co_eff2, var_matrix2, converged = fit3(vel,spec,guess2,bound2,noise)	
		if(n_peaks==3):
			r_chi_sq2, res2, co_eff2, var_matrix2, converged = fit4(vel,spec,guess2,bound2,noise)	
		if(n_peaks==4):
			r_chi_sq2, res2, co_eff2, var_matrix2, converged = fit5(vel,spec,guess2,bound2,noise)	
		if(n_peaks==5):
			r_chi_sq2, res2, co_eff2, var_matrix2, converged = fit6(vel,spec,guess2,bound2,noise)	


		if(debug==1):

			print "##### Poor fit so we add one #####"
			print "Fitted with ", n_peaks + 1, " peaks"
			print "Reduced chi_sq = ", r_chi_sq2
			print "Residuals = ", res2
			print "Co-effs = ", co_eff2
			print "Converged = ", converged
			print " "
			

		#Check if the fit converged we return the old bad fit.
		if(converged==0):

			return co_eff,r_chi_sq

		### Check the new fit for overlapping peaks
		if(overlap_tag==1):
			overlap = check_overlap(co_eff2,dv)

		### Check for tiny and thin components, this normally comes about from fitting noise
		t_and_t = check_tiny_and_thin(co_eff2,dv,n,noise)

		### if we have overlapping peaks or the new fit is worse than the old then we output the previous fit or we have a tiny and thin component. We therefore keep the old fit
		if(overlap==1 or r_chi_sq2 > r_chi_sq or t_and_t==1):

			return co_eff, r_chi_sq

		### If the new fit doesn't contain an overlap and isn't overfitting and is a better fit then we output the new fit.
		elif(overlap==0 and r_chi_sq2 < r_chi_sq and t_and_t==0):	

			return co_eff2, r_chi_sq2			


	## No overlap and the fit is good enough so lets output it
	if(overlap==0 and r_chi_sq<chi_limit):

		return co_eff,r_chi_sq

	if(overlap==0 and r_chi_sq>chi_limit and n_peaks==6):

		return co_eff,r_chi_sq



##### Test run functions, used to help determine the optimal values for the 3 important parameters: chi_limit, smoothing_length and signal to noise ratio


### The first is for the single Gaussian test, used to determine the average error on the fitting parameters

def single_gaussian_test(paramfile):

	param = ReadParameters(paramfile)

	### Set off the lower integrated emission limit variable to 0 in case it was set to none zero in the parameter file

	if(param["lower_integrated_emission_limit"] !=0):
		param["lower_integrated_emission_limit"] = 0.0

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

	print "#######################"
	print "#### Test progress ####"
	print "#######################"
	print " "

	for ii in range(0,num_test):

		if((ii+1)%numpy.int(num_test/10.)==0):
			print "The test is %.2f %% complete" %(100*(ii+1)/numpy.float(num_test))

		### Pick a random amplitude, centroid and width
		amp = numpy.random.rand(1)*(amp_max - amp_min) + amp_min
		cen = numpy.random.rand(1)*(cen_max - cen_min) + cen_min
		width = numpy.random.rand(1)*(wid_max - wid_min) + wid_min

		### Produce Gaussian and add noise
		y = gaussone(v,amp[0],cen[0],width[0])
		no = numpy.random.normal(loc=0.0,scale=noise_level,size=spec_nv)
		y = y+no


		### Fit this synthetic spectrum
		c2 = fit_single_line(v,y,param)

		co=c2[0]
		chi = c2[1]

		### Check that a fit was found
		if(co[0]==0 or co[0]<0):
			n = n + 1
			print "No line was detected"
			print "The amplitude, centroid and width was: ", amp, cen, width	

		### If there was a fit, check for the number of components found
		else:

			### If only a single Gaussian found, store the errors on the parameters
			if(len(co)==3):

				a_e.append(numpy.fabs(amp-co[::3])/amp)
				c_e.append(numpy.fabs((cen-co[1::3])/cen))
				w_e.append(numpy.fabs(width-co[2::3])/width)
				c_chi.append(chi)


				### If the plot tag is turned on, plot the input parameter against the fitted parameters
				if(plot_tag==1):
					matplotlib.pyplot.figure(1)
					matplotlib.pyplot.plot(amp,co[0],"kx")
					matplotlib.pyplot.figure(2)
					matplotlib.pyplot.plot(cen,co[1],"kx")
					matplotlib.pyplot.figure(3)
					matplotlib.pyplot.plot(width,co[2],"kx")


			### More than one component was detected.
			else:
				n2 = n2 + 1
				print "More than one component was detected"
				print "The input components were: ", amp, cen, width
				print "The output amplitudes, centroids and widths were: ", co[0::3], co[1::3], co[2::3]


	### Print out the median errors and the interquartile range
	a_e = 100*numpy.array(a_e)
	c_e = 100*numpy.array(c_e)
	w_e = 100*numpy.array(w_e)

	print " "
	print " "
	print " "
	print "##############################################"
	print "#### Results for the single Gaussian test ####"
	print "##############################################"

	print "Median error on amplitude as a percentage = %.2f + %.2f - %.2f" %(numpy.percentile(a_e, 50), (numpy.percentile(a_e,75)-numpy.percentile(a_e,50)), (numpy.percentile(a_e,50)-numpy.percentile(a_e,25)))
	print "Median error on centroids as a percentage = %.2f + %.2f - %.2f" %(numpy.percentile(c_e, 50), (numpy.percentile(c_e,75)-numpy.percentile(c_e,50)), (numpy.percentile(c_e,50)-numpy.percentile(c_e,25)))
	print "Median error on widths as a percentage    = %.2f + %.2f - %.2f" %(numpy.percentile(w_e, 50), (numpy.percentile(w_e,75)-numpy.percentile(w_e,50)) , (numpy.percentile(w_e,50)-numpy.percentile(w_e,25)))

	### Also show the reduced chi_squared for these fits
	c_chi = numpy.array(c_chi)
	print "Median reduced chi_squared                = %.2f + %.2f - %.2f" %(numpy.percentile(c_chi, 50), (numpy.percentile(c_chi,75)-numpy.percentile(c_chi,50)) , (numpy.percentile(c_chi,50)-numpy.percentile(c_chi,25)))


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
		matplotlib.pyplot.figure(4)
		matplotlib.pyplot.hist(c_chi,bins=40)
		matplotlib.pyplot.show()


	return



### Here we now start the multiple component test. This is to test the code's ability to determine the number of components in test spectra.

def multi_gaussian_test(paramfile):

	param = ReadParameters(paramfile)

	### Set off the lower integrated emission limit variable to 0 in case it was set to none zero in the parameter file

	if(param["lower_integrated_emission_limit"] !=0):
		param["lower_integrated_emission_limit"] = 0.0

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

	### Construct the velocity array

	v = numpy.linspace(spec_min,spec_max,spec_nv)

	### Initiize the seed for the random number generators
	numpy.random.seed(4)

	### arrays to store the input number of peaks and the output number of peaks
	ni = numpy.zeros(num_test)
	nf = numpy.zeros(num_test)

	count = 0

	print "#######################"
	print "#### Test progress ####"
	print "#######################"
	print " "

	for ii in range(0,num_test):

		if((ii+1)%numpy.int(num_test/10.)==0):
			print "The test is %.2f %% complete" %(100*(ii+1)/numpy.float(num_test))

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


		### Now we have the parameters for the test spectra, we create them and add noise.
		if(npeak==1):
			y = gaussone(v,amp[0],cen[0],width[0])

		if(npeak==2):
			y = gausstwo(v,amp[0],cen[0],width[0],amp[1],cen[1],width[1])

		if(npeak==3):
			y = gaussthree(v,amp[0],cen[0],width[0],amp[1],cen[1],width[1],amp[2],cen[2],width[2])

		if(npeak==4):
			y = gaussfour(v,amp[0],cen[0],width[0],amp[1],cen[1],width[1],amp[2],cen[2],width[2],amp[3],cen[3],width[3])


		no =  numpy.random.normal(loc=0.0,scale=noise_level,size=spec_nv)
		y = y+no

		### Fit this synthetic spectrum
		c2 = fit_single_line(v,y,param)

		co=c2[0]
		chi = c2[1]

		### Check that a fit was found
		if(co[0]==0 or co[0]<0):
			nf[ii] = 0
			print "No components detected"
			print "The amplitude, centroid and width were: ", amp, cen, width
			

		### If a fit was found then determine the number of components and find it.
		else:
			if(len(co)==3):
				nf[ii] = 1

			if(len(co)==6):
				nf[ii] = 2

			if(len(co)==9):
				nf[ii] = 3

			if(len(co)==12):
				nf[ii] = 4


		### If the number of fitted components is not equal to the input number store this information.
		if(ni[ii]!=nf[ii]):
			count = count+1

	print " "
	print " "
	print " "
	print "################################################"
	print "#### Results for the multiple Gaussian test ####"
	print "################################################"

	print "%d test spectra were fitted with the wrong number of components out of %d tests" %(count,len(ni[ni>0]))
	print "Thus the code has a %.2f %% success rate for these parameters" %(100*(1-count/numpy.float(len(ni[ni>0]))))

	return

##### Function for reading parameter file

def ReadParameters(param_file):


	### The dictionaries for the type of variable and the variable itself

	type_of_var = {"chi_limit"                        :   "float",
                      "check_overlap"                     :   "int",
                      "debug"                             :   "int",
                      "lower_integrated_emission_limit"   :   "float",
                      "smoothing_length"                  :   "float",
	              "variable_noise"                    :   "int",
	              "noise_level"                       :   "float",
	              "noise_clip"                        :   "int",
	              "signal_to_noise_ratio"             :   "float",
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
                      "in_file_name"                      :   "str",
                      "out_file_base"                     :   "str"}

	param = {"chi_limit"                        :   1.5,
                "check_overlap"                     :   1,
                "debug"                             :   0,
                "lower_integrated_emission_limit"   :   0.5,
                "smoothing_length"                  :   3.0,
	        "variable_noise"                    :   0,
	        "noise_level"                       :   0.1,
	        "noise_clip"                        :   50,
	        "signal_to_noise_ratio"             :   5,
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
                "in_file_name"                      :  "input.fits",
                "out_file_base"                     :  "output"}


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

					print "The variable is neither a string, float or integer. I don't know how to deal with this"

			except KeyError:

				print "There is no such parameter. Add it to the type_of_var and param dictionaries"


		f.close()

	### Print the parameters to screen

	print " "
	print " "
	print " "
	print "############################################"
	print "################ Parameters ################"
	print "############################################"
	print ""
	print "############# Important three ##############"
	print "Smoothing length                = ", param["smoothing_length"]
	print "Reduced Chi_sqaure limit        = ", param["chi_limit"]
	print "Signal to noise ratio           = ", param["signal_to_noise_ratio"]
	print " "

	print "######## Non-test noise parameters #########"
	print "Variable noise                  = ", param["variable_noise"]
	if(param["variable_noise"]==0):
		print "Constant noise level            = ", param["noise_level"]
	elif(param["variable_noise"]==1): 
		print "Number of bins used for noise   = ", param["noise_clip"]
	print "Integrated emission lower limit = ", param["lower_integrated_emission_limit"]
	print " "

	print "################## Flags ###################"
	print "Check overlap                   = ", param["check_overlap"]
	print "Debug switch                    = ", param["debug"]
	print " "

	print "############# Fits file fitting ############"
	print "Input Fits file name            = ", param["in_file_name"]
	print "Output name base                = ", param["out_file_base"]
	print " "

	print "######### Parameters for test runs #########"
	print "Number of test spectra          = ", param["test_number"]
	print "Spectral range                  = [", param["test_spec_min"],",",param["test_spec_max"],"]"
	print "Number of spectral bins         = ", param["test_spec_num"]
	print "Test noise level                = ", param["test_noise"]
	print "Amplitude range of components   = [", param["test_amplitude_min"],",",param["test_amplitude_max"],"]"
	print "Velocity centroid range         = [", param["test_vel_cen_min"],",",param["test_vel_cen_max"],"]"
	print "Width range of components       = [", param["test_width_min"],",",param["test_width_max"],"]"
	print "Plotting switch                 = ", param["test_plot_tag"]
	print " "

	print " "
	print " "

	return param







##### Function which fits a whole fits cube if the fits cube has all the right information in the header

def fit_a_fits(param_file):

	### Read the parameter file and get the input fits file name

	param = ReadParameters(param_file)
	fitsfile = param["in_file_name"]
	
	### open the fits file with astropy and extract the ppv datacube

	spec_fits = astropy.io.fits.open(fitsfile)
	ppv_data = spec_fits[0].data

	### check to make sure the datacube is 3D

	if(numpy.shape(ppv_data)[0] == 1):
		ppv_data = ppv_data[0,:,:,:]

	### Extract the required information from the fits file header to construct the velocity axis

	refnv = numpy.int(spec_fits[0].header["CRPIX3"])
	refv = numpy.double(spec_fits[0].header["CRVAL3"])
	dv = numpy.double(spec_fits[0].header["CDELT3"])
	nv = len(ppv_data[:,0,0])

	### Construct the velocity axis 

	vel = numpy.zeros(nv)
	for ii in range(0,nv):
		vel[ii] = refv + (ii-refnv+1)*dv

	neg_dv = 0

	if(dv<0):

		vel = numpy.flip(vel,axis=0)
		ppv_data = numpy.flip(ppv_data,axis=0)
		dv = numpy.fabs(dv)
		neg_dv = 1

	minv = vel[0]
	maxv = vel[nv-1]

	### Get the size of the ppv datacube and allocate the arrays for the output.

	nx = len(ppv_data[0,0,:])
	ny = len(ppv_data[0,:,0])

	amp_out = numpy.zeros_like(ppv_data)
	cen_out = numpy.zeros_like(ppv_data)
	wid_out = numpy.zeros_like(ppv_data)
	chi_out = numpy.zeros_like(ppv_data)

	### Just counters

	count = numpy.zeros(6)
	no_con = 0


	print "##########################"
	print "#### Fitting progress ####"
	print "##########################"

	### Loop over the lines of sight in the datacube and try to fit each one

	for ii in range(0,ny):
		if((ii+1)%numpy.int(ny/10.)==0):
			print "We are %.2f %% of the way through the cube" %(100*(ii+1)/numpy.float(ny))
		for jj in range(0,nx):

			### Take the spectrum from the line of sight

			spec = ppv_data[:,ii,jj]

			### Make sure there are no nans in the spectrum

			nan_check = numpy.sum(numpy.isnan(spec))
			if(nan_check>0):
				continue

			### Fit spectrum and unpack the outputs

			co = fit_single_line(vel,spec,param)
			co_eff = co[0]
			rc2 = co[1]

			### Check for no convergence or if no peak was detected.			

			if(co_eff[0] == 0):
				no_con = no_con + 1
				continue
			if(co_eff[0] < 0):
				continue

			### If there was a fit, work out the number of components fitted and store this number in the count array	

			n = len(co_eff[::3])
			count[n-1] = count[n-1] + 1

			### Loop over the components found and store in the 4 output arrays

			for kk in range(0,n):

				index_v = int((co_eff[3*kk+1]-minv)/dv)	

				if(index_v>=0 and index_v < nv):				
					amp_out[index_v,ii,jj] = co_eff[3*kk]
					cen_out[index_v,ii,jj] = co_eff[3*kk + 1]	
					wid_out[index_v,ii,jj] = co_eff[3*kk + 2]
					chi_out[index_v,ii,jj] = rc2



	### Print out results

	print " "
	print " "
	print " "
	print "#########################"
	print "###### Fit results ######"
	print "#########################"
	print "There are %d spectra with 1 component, %.2f %% of total fitted spectra" %(count[0], 100*numpy.float(count[0])/numpy.sum(count))
	print "There are %d spectra with 2 component, %.2f %% of total fitted spectra" %(count[1], 100*numpy.float(count[1])/numpy.sum(count))
	print "There are %d spectra with 3 component, %.2f %% of total fitted spectra" %(count[2], 100*numpy.float(count[2])/numpy.sum(count))
	print "There are %d spectra with 4 component, %.2f %% of total fitted spectra" %(count[3], 100*numpy.float(count[3])/numpy.sum(count))
	print "There are %d spectra with 5 component, %.2f %% of total fitted spectra" %(count[4], 100*numpy.float(count[4])/numpy.sum(count))
	print "There are %d spectra with 6 component, %.2f %% of total fitted spectra" %(count[5], 100*numpy.float(count[5])/numpy.sum(count))
	print " "
	if(no_con>0):
		print "There was no convergence for %d spectra" %no_con


	### Set up output. Copy the header from the input file and use that for the output files

	output_base = param["out_file_base"]
	hdu = astropy.io.fits.PrimaryHDU()
	hdu.header = spec_fits[0].header

	if(neg_dv==1):
		amp_out = numpy.flip(amp_out,axis=0)
		cen_out = numpy.flip(cen_out,axis=0)
		wid_out = numpy.flip(wid_out,axis=0)
		chi_out = numpy.flip(chi_out,axis=0)

	### Output amplitudes

	hdu.data = amp_out
	outfile = "Amp_"+output_base+".fits"
	if(os.path.isfile(outfile)):
		os.remove(outfile)
	hdu.writeto(outfile)

	### Output widths

	hdu.data = wid_out
	outfile = "Width_"+output_base+".fits"
	if(os.path.isfile(outfile)):
		os.remove(outfile)
	hdu.writeto(outfile)

	### Output reduced chi_square

	hdu.data = chi_out
	outfile = "Chi_"+output_base+".fits"
	if(os.path.isfile(outfile)):
		os.remove(outfile)
	hdu.writeto(outfile)

	### Output centoids

	hdu.data = cen_out
	outfile = "Vel_"+output_base+".fits"
	if(os.path.isfile(outfile)):
		os.remove(outfile)
	hdu.writeto(outfile)

	return


##### The functions that are used for fitting. More than 6 Gaussians can be added.

def gaussone(x, a, x0, sigma):
    return a*numpy.exp(-(x-x0)**2/(2*sigma**2))

def gausstwo(x, a, x0, sigma, a2, x02, sigma2):
    return a*numpy.exp(-(x-x0)**2/(2*sigma**2)) + a2*numpy.exp(-(x-x02)**2/(2*sigma2**2)) 

def gaussthree(x, a, x0, sigma, a2, x02, sigma2,a3,x03,sigma3):
    return a*numpy.exp(-(x-x0)**2/(2*sigma**2)) + a2*numpy.exp(-(x-x02)**2/(2*sigma2**2))+ a3*numpy.exp(-(x-x03)**2/(2*sigma3**2))

def gaussfour(x, a, x0, sigma, a2, x02, sigma2,a3,x03,sigma3,a4,x04,sigma4):
    return a*numpy.exp(-(x-x0)**2/(2*sigma**2)) + a2*numpy.exp(-(x-x02)**2/(2*sigma2**2))+ a3*numpy.exp(-(x-x03)**2/(2*sigma3**2)) + a4*numpy.exp(-(x-x04)**2/(2*sigma4**2))

def gaussfive(x, a, x0, sigma, a2, x02, sigma2,a3,x03,sigma3,a4,x04,sigma4,a5,x05,sigma5):
    return a*numpy.exp(-(x-x0)**2/(2*sigma**2)) + a2*numpy.exp(-(x-x02)**2/(2*sigma2**2))+ a3*numpy.exp(-(x-x03)**2/(2*sigma3**2)) + a4*numpy.exp(-(x-x04)**2/(2*sigma4**2)) + a5*numpy.exp(-(x-x05)**2/(2*sigma5**2))

def gausssix(x, a, x0, sigma, a2, x02, sigma2,a3,x03,sigma3,a4,x04,sigma4,a5,x05,sigma5,a6,x06,sigma6):
    return a*numpy.exp(-(x-x0)**2/(2*sigma**2)) + a2*numpy.exp(-(x-x02)**2/(2*sigma2**2))+ a3*numpy.exp(-(x-x03)**2/(2*sigma3**2)) + a4*numpy.exp(-(x-x04)**2/(2*sigma4**2)) + a5*numpy.exp(-(x-x05)**2/(2*sigma5**2)) +a6*numpy.exp(-(x-x06)**2/(2*sigma6**2))






### Checking functions.

### Check for the low amplitude and very thin gaussian fits. These typically appear when there is one too many peaks detected.
def check_tiny_and_thin(co_eff,dv,n,noise):

	tt=0

	amp = co_eff[::3]
	wid = co_eff[2::3]

	for ii in range(0,len(amp)):
		
		if(amp[ii]<1.01*n*noise and wid[ii]<1.01*dv):
			tt=1

	return tt

### Check if two components lie within one velocity bin of each other. This check can be turned on or off.
def check_overlap(co_eff,dv):

	overlap = 0

	v_cent = co_eff[1::3]
	for kk in range(0,len(v_cent)-1):
		for ll in range(kk+1,len(v_cent)):
			if(v_cent[ll] < v_cent[kk] + dv and v_cent[ll] > v_cent[kk] - dv):
				overlap = 1

	return overlap










##### Wrappers for the fits for different number of peaks

def fit1(vel,spec,guess,bound,noise):

	try:
		co_eff, var_matrix = curve_fit(gaussone,vel,spec,p0=guess,method="trf",bounds=bound)

		chi_sq = sum((spec-gaussone(vel,*co_eff))**2) / noise**2
		r_chi_sq = chi_sq / (len(spec) - 3)

		fit = gaussone(vel,*co_eff)
		res = spec-fit
		converged = 1

	except RuntimeError:
		converged = 0
		r_chi_sq = 0
		res = 0
		co_eff = 0
		var_matrix = 0

	return r_chi_sq, res, co_eff, var_matrix, converged

def fit2(vel,spec,guess,bound,noise):
	
	try:
		co_eff, var_matrix = curve_fit(gausstwo,vel,spec,p0=guess,method="trf",bounds=bound)

		chi_sq = sum((spec-gausstwo(vel,*co_eff))**2) / noise**2
		r_chi_sq = chi_sq / (len(spec) - 6)

		fit = gausstwo(vel,*co_eff)
		res = spec-fit
		converged = 1

	except RuntimeError:
		converged = 0
		r_chi_sq = 0
		res = 0
		co_eff = 0
		var_matrix = 0

	return r_chi_sq, res, co_eff, var_matrix, converged

def fit3(vel,spec,guess,bound,noise):

	try:
		co_eff, var_matrix = curve_fit(gaussthree,vel,spec,p0=guess,method="trf",bounds=bound)

		chi_sq = sum((spec-gaussthree(vel,*co_eff))**2) / noise**2
		r_chi_sq = chi_sq / (len(spec) - 9)

		fit = gaussthree(vel,*co_eff)
		res = spec-fit
		converged = 1

	except RuntimeError:
		converged = 0
		r_chi_sq = 0
		res = 0
		co_eff = 0
		var_matrix = 0

	return r_chi_sq, res, co_eff, var_matrix, converged

def fit4(vel,spec,guess,bound,noise):

	try:
		co_eff, var_matrix = curve_fit(gaussfour,vel,spec,p0=guess,method="trf",bounds=bound)

		chi_sq = sum((spec-gaussfour(vel,*co_eff))**2) / noise**2
		r_chi_sq = chi_sq / (len(spec) - 12)

		fit = gaussfour(vel,*co_eff)
		res = spec-fit
		converged=1

	except RuntimeError:
		converged = 0
		r_chi_sq = 0
		res = 0
		co_eff = 0
		var_matrix = 0

	return r_chi_sq, res, co_eff, var_matrix, converged

def fit5(vel,spec,guess,bound,noise):

	try:
		co_eff, var_matrix = curve_fit(gaussfive,vel,spec,p0=guess,method="trf",bounds=bound)

		chi_sq = sum((spec-gaussfive(vel,*co_eff))**2) / noise**2
		r_chi_sq = chi_sq / (len(spec) - 15)

		fit = gaussfive(vel,*co_eff)
		res = spec-fit
		converged = 1

	except RuntimeError:
		converged = 0
		r_chi_sq = 0
		res = 0
		co_eff = 0
		var_matrix = 0

	return r_chi_sq, res, co_eff, var_matrix, converged

def fit6(vel,spec,guess,bound,noise):

	try:
		co_eff, var_matrix = curve_fit(gausssix,vel,spec,p0=guess,method="trf",bounds=bound)

		chi_sq = sum((spec-gausssix(vel,*co_eff))**2) / noise**2
		r_chi_sq = chi_sq / (len(spec) - 18)

		fit = gausssix(vel,*co_eff)
		res = spec-fit
		converged = 1

	except RuntimeError:
		converged = 0
		r_chi_sq = 0
		res = 0
		co_eff = 0
		var_matrix = 0

	return r_chi_sq, res, co_eff, var_matrix, converged
