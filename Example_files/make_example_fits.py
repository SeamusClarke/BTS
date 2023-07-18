from pylab import *
from astropy.io import fits

cube = zeros((300,100,100))

x0 = 30
y0 = 30
v0 = 250
s0 = 10
r0 = 20

x1 = 60
y1 = 60
v1 = 190
s1 = 15
r1 = 30

v = arange(0,300,1)
print(v)

for ii in range(0,100):
	for jj in range(0,100):

		if(sqrt((x0-ii)**2 + (y0-jj)**2)<r0):
			cube[:,ii,jj] = cube[:,ii,jj] + 2.*exp(-(v-v0)**2 / (2*s0**2))

		if(sqrt((x1-ii)**2 + (y1-jj)**2)<r1):
			cube[:,ii,jj] = cube[:,ii,jj] + 1.*exp(-(v-v1)**2 / (2*s1**2))


noise_free = copy(cube)
cube = cube + normal(loc=0.0,scale=0.2,size=(300,100,100))


hdu = fits.PrimaryHDU()
hdu.data = cube 

hdu.header["NAXIS3"] = 300
hdu.header["CTYPE3"] = "VRAD"
hdu.header["CRPIX3"] = 0
hdu.header["CRVAL3"] = 0
hdu.header["CDELT3"] = 1


hdu.writeto("example_cube.fits",overwrite=True)
