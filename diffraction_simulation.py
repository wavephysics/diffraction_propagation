# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 09:57:30 2021

@author: OmarBoughdad
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from numba import njit
from scipy.ndimage import convolve
from scipy.fftpack import fft2, ifft2, fftshift, fftfreq
from scipy.special import comb, factorial
from scipy.signal import convolve2d, convolve
from PIL import Image, ImageDraw, ImageFont


arial = 'arial.ttf'
coop_black = 'COOPBL.TTF'


# DEFINE THE UNITS


mm = 1e-3           # Dimensions
cm = mm*1e1         
um = mm*mm
nm = um*mm


class DiffractionSolver(object):
    '''
    The following Class is considered to solve the problem of diffraction due to an aperture.
    To do so we will make use of the Fresnel Faunhoffer integral formulas, the angular 
    spectrum method, the direct integration method and the beam propagation method. All these
    methods consider the paraxial approximation of the light beam [1-3] (see the biblio below).
    We will place ourselves in the frame of this approximation to simulate the propagation
    of light after diffraction.
    
    [1] https://www.cambridge.org/core/books/elements-of-nonlinear-optics/F6B3C66E6115CD3DE8F615DF16BBB47C  (pp 309-311)
    [2] https://physics.stackexchange.com/questions/290152/difference-between-the-paraxial-approximation-and-the-fresnel-approximation
    [3] https://www.osapublishing.org/josaa/abstract.cfm?uri=josaa-13-9-1816
    
    The Class DiffractionSolver has many functions devided into three cathegories. The first is to define the physical
    and the Fourier spaces of the system. The second one considers defining the aperture source.
    Note that, when studying diffraction through an aperture, there is no need to define the wavefunction
    before the aperture. Because, we consider a plane wave hiting the aperture. Therfore, the intensity 
    of light is uniform at the aperture. Let us stress that the beam propagates along the z-direction 
    the position of the aperture is z0 = 0. The third category of functions is the solver category.

    '''
    def __init__(self, p=None, Lx=None, Ly=None, lbd=None, pixel_size_x=None, pixel_size_y=None, z=None, zstep=None, prop_method=None):
        
        ''' 
        The following funtion enables to define the physical
        and the fourier space of the system. 
        
        Inputs :
        
            - p: is an integer defining the grid points (size)
            (integer no unit)
        
            - Lx and Ly are the dimesions of the real space. Litteraly,
            they refer to the size of the simulation window (meters)
        
            - lbd is the wavelength of the light beam (meters)
            
            - pixel_size_x : the size of the pixel along x
            
            - pixel_size_y : the size of the pixel along y
            
            - z : Distance of propagation along z-axis
        
            Note that if you provide the dimensions of the system Lx and Ly
            please keep the pixel size None. Because, it is calculated 
            using the dimensions and the number of grid point given
        
            If you prove the pixel sizes (x and y). The dimension of the
            system are calculated based on that.
        '''
        
        if p == None:               # Avoid None value
            self.p = 11                  # Assign 11 to the integer
        else:
            self.p = p
        
        if lbd == None:
            self.lbd = 650*nm
        else:
            self.lbd = lbd
            
        if z == None:
            self.z = 50*cm
        else:
            self.z = z
            
        if zstep == None:
            self.zstep = 20
        else:
            self.zstep = zstep
                
        self.k0 = (2*np.pi)*(self.lbd)**-1            # Wavenumber of the light beam along z-axis
        self.nx = 2**self.p                   # Number of grid points along x-axis
        self.ny = 2**self.p                   # Number of grid points along y-axis
        self.Mx = self.nx + 1
        self.My = self.ny + 1
        self.xn = np.linspace(-int(self.Mx/2) , int(self.Mx/2), self.Mx)               # Define vector of points along x-axis
        self.yn = np.linspace(-int(self.My/2) , int(self.My/2), self.My)               # Define vector of points along y-axis
        self.dz = self.z/self.zstep
        # self.aperture_radius = 0
        
        
        # The value Lx, Ly, pixel_size_x and pixel_size_y may go into a conflict
        # Because theya re linearly related. That is why we shoud define a condition
        # on both axis x and y.
        
        if Lx == None:
            if pixel_size_x == None:
                self.Lx = 12*cm                                     #                        
                self.pixel_size_x = self.Lx/self.nx                 #
            else:                                                   # To Define Lx based on pixel_size_x or the inverse
                self.pixel_size_x = pixel_size_x                    #
                self.Lx = self.pixel_size_x*self.nx                 #
                
        if Ly == None:                                              #
            if pixel_size_y == None:                                #
                self.Ly = 12*cm                                     #
                self.pixel_size_y = self.Ly/self.ny                 # To Define Ly based on pixel_size_y or the inverse
            else:                                                   #
                self.pixel_size_y = pixel_size_y                    #
                self.Ly = self.pixel_size_y*self.ny                 #
    
        if pixel_size_x == None:                                    #
            if Lx == None:                                          #
                self.pixel_size_x = 20*nm                           #
                self.Lx = self.pixel_size_x*self.nx                 # To Define Lx based on pixel_size_x or the inverse
            else:                                                   #
                self.Lx = Lx                                        #
                self.pixel_size_x = self.Lx/self.nx                 #
                
        if pixel_size_y == None:                                    #
            if Ly == None:                                          #
                self.pixel_size_y = 20*nm                           #
                self.Ly = self.pixel_size_y*self.ny                 # To Define Ly based on pixel_size_y or the inverse
            else:                                                   #
                self.Ly = Ly                                        #
                self.pixel_size_y = self.Ly/self.ny                 #
                
        self.dx = self.pixel_size_x                                 #
        self.dy = self.pixel_size_y                                 #
                
        self.x = self.xn*self.pixel_size_x                          # Defines the space x-vector
        self.y = self.yn*self.pixel_size_y                          # Defines the space y-vector
        self.X, self.Y = np.meshgrid(self.x, self.y)                # creates the mesh of the system
        
        # The part below enables to define the fourier space of the system.
        # There is a function in the numpy library dedicated to defining
        # the fourier space. It is denoted by fftfreq. The function takes 
        # the number of grid point as an arguement and the spacing between
        # the difference frequencies.
        # There is another way to define the Fourier space variables, namely,
        # kx and ky. dkx = pi/Lx and dky = pi/Ly
        
        self.dkx = np.pi/(self.Lx)                      # Calculates the pixel size in the fourier space x-axis
        self.dky = np.pi/(self.Ly)                      # Calculates the pixel size in the fourier space y-axis
        
        self.kx = np.linspace(-(self.dkx / 2 ) * self.Mx, self.dkx / 2 * self.Mx, self.Mx)
        self.ky = np.linspace(-(self.dky / 2 ) * self.My, self.dky / 2 * self.My, self.My)
        
        # self.kx = np.fft.fftfreq(self.Mx, d=self.dx)              # Predfined function to calculate the spatial frequency along x
        # self.ky = np.fft.fftfreq(self.My, d=self.dy)              # Predfined function to calculate the spatial frequency along y
        self.kx.sort(); self.ky.sort()                                # Sorts the spatial frequency domain
        self.kX, self.kY = np.meshgrid(self.kx, self.ky)              # Grid 
        
        if prop_method == None:
            self.prop_method = 'frsnl_frnhfer'
        else:
            self.prop_method =  prop_method
        
        print('\n')
        print('\n')
        
        print('|-------------------------------------------------------------------------------------------------|')
        print('|                                  THE PARAMETERS OF THE SYSTEM                                   |')
        print('|-------------------------------------------------------------------------------------------------|')
        print('|                         Parameter                             |               Value             |')
        print('|-------------------------------------------------------------------------------------------------|')
        # print('\n ')
        print('| Number of grid points along x and y                           |                     ', (self.Mx, self.My),'|')
        print('|-------------------------------------------------------------------------------------------------|')
        print('| Wavelength                                                    |                 ', self.lbd,'in (m) |')
        print('|-------------------------------------------------------------------------------------------------|')
        print('| Wavenumber                                                    |      ', self.k0,'in (m-1) |')
        print('|-------------------------------------------------------------------------------------------------|')
        print('| Pixel size along x-axis                                       |  ', self.dx,  'in (m) |')
        print('|-------------------------------------------------------------------------------------------------|')
        print('| Pixel size along y-axis                                       |  ', self.dy, 'in (m) |')
        print('|-------------------------------------------------------------------------------------------------|')
        print('| Physical dimension of the system along x                      |    ', self.Lx, 'in (m) |')
        print('|-------------------------------------------------------------------------------------------------|')
        print('| Physical dimension of the system along y                      |    ', self.Lx, 'in (m) |')
        print('|-------------------------------------------------------------------------------------------------|')
        print('| Pixel size along kx-axis  (Fourier space)                     |    ', self.dkx,'in (m-1) |')
        print('|-------------------------------------------------------------------------------------------------|')
        print('| Pixel size along ky-axis  (Fourier space)                     |    ', self.dky,'in (m-1) |')
        print('|-------------------------------------------------------------------------------------------------|')
        print('| Distance of propagation                                       |                     ', self.z,'in (m) |')
        print('|-------------------------------------------------------------------------------------------------|')
        print('| Number of steps along z                                       |                             ', self.zstep,'|')
        print('|-------------------------------------------------------------------------------------------------|')
        print('| Step of proagation                                            |                   ', self.dz,'in (m) |')
        print('|-------------------------------------------------------------------------------------------------|')
        
        print('\n')



        
    def intensity(self, field):
        '''
        Function that calculates the intensity of the light beam. By measuring the light 
        using a camera, we measure it intensity.
        
        Input:
            
            Field matrix of shape (Mx, My)
        
        Output 
            Intensity of the light beam of the same shape    
        '''
        return np.real(field*np.conj(field))

    def ft(self, field):
        '''
        Calculates the fast fourier transform of the field and applies an fftshift to
        center the central frequency
        
        Input:
            
            Field matrix of shape (Mx, My)
        
        Output 
            Fourier transform of the field matrix of shape (Mx, My)
        '''
        
        return fftshift(fft2(field))
        
    def ift(self, field):
        '''
        Calculates the Inverse fast fourier transform of the field 
        
        Input:
            
            Field matrix of shape (Mx, My)
        
        Output 
            Fourier transform of the field matrix of shape (Mx, My)
        '''
        return ifft2(field)
    
    def fft_convolve(self, field1, field2):
        '''
        The following function performs the convolution of two 2D functions using the
        Fourier transform. One of the fastest way to achieve convolution is to use the FFT
        
        
        Input:
            
            2 Fields matrix of shape (Mx, My)
        
        Output 
            Convolution product of the two matrices
        
        '''
        ft_field1 = self.ft(field1)
        ft_field2 = self.ft(field2)
        m,n = ft_field1.shape
        cc = np.real(  self.ift( ft_field1*ft_field2 )   )
        cc = np.roll(cc, -m/2+1,axis=0)
        cc = np.roll(cc, -n/2+1,axis=1)
        return cc
    

    def convolve(self, field1, field2, padding=0, strides=1):
        '''
        The following function enables to calculate the convolution product of two matrices
        of diffrerent sizes. Unlike the previous function, the convolution will be done wihtout
        using the fourier transform. This normally should take more time in calculation.
        
        Input:
            
            - 2 fields matrix of shape (Mx, My). However, they could have different shapes
            - Padding Integer refering to if we would like to pad the image before the convolution
            - Stride integer refering to the step of the convolution in case the kernel (field 2) has not
            the same shape of the field 1
            
        Output:
            
            Conv_product matrix of shape ....
        '''
        
        field2 = np.flipud( np.fliplr( field2 ) )
        
        x_shape_field2 = field2.shape[0]                                #               
        y_shape_field2 = field2.shape[1]                                #  Shapes of the field
        
        x_shape_field1 = field1.shape[0]                                #
        y_shape_field1 = field1.shape[1]                                #
        
        x_shape_output = int(((x_shape_field1 - x_shape_field2 + 2 * padding) / strides) + 1)
        y_shape_output = int(((y_shape_field1 - y_shape_field2 + 2 * padding) / strides) + 1)
        
        conv_product = np.zeros((x_shape_output, y_shape_output))
        
        if padding != 0:
            padded_field = np.zeros(( x_shape_field1 + padding**2,  y_shape_field1 + padding**2))
            padded_field[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
            print(imagePadded)
        else:
            padded_field = field1
        
        for j in range( y_shape_field1 ):
            if j > y_shape_output - y_shape_field2:
                break
            if jj % strides == 0:
                for ii in range( x_shape_field1 ):
                    if ii > x_shape_field1 - x_shape_field2:
                        break
                    try:
                        if ii % strides == 0 :
                            conv_product[ii,jj] = (field2 * padded_field[ii: ii + x_shape_field2, jj: jj + y_shape_field2]).sum()
                    except:
                        break
        return conv_product


        
    def aperture_source(self, a=300, b=300, poly_bound_circle=[None, None, 200], poly_n_side=3,\
                        poly_rotation=0, x_r=None, y_r=None, geometry='circ_aperture',\
                        text_to_draw=None, text_font_size=50):
        '''
        The following function is used to generate the waveform at the position z0
        when a plane wave hits the diffraction plane. The diffraction plane could 
        take many forms, namely, circular, ractangular, polygon even a text form.
        In fact, we can write a text (any word) and see how diffraction occurs.
        To do so, we make use of a predefined library called Pillow (PIL).
        The library enables us to create different forms (drawings)
        
        the function takes a few arguments of different types
        
        - a : The major axis of the ellipse, the  length of the rectangle (integer in pixels)
        - b : The minor axis of the ellipse, the width of the rectangle (integer in pixels)
        - poly_bound_circle : (..,..,..) tuple of three elements, first 2 define
        the position of the centre and the last one defines the radius of the circle
        embedding the polygone (float, float, float)
        - poly_n_side : The number of sides of the polygone (integer)
        - poly_rotation : The rotation applied to the polygone (float values)
        - x_r,  y_r : the position of the centre of the rectangle the circle and
        the polygone (float value)
        - gerometry : a string to precise the fom of the aperture, takes 
        "circ_aperture", "rect_aperture", "regular_polygon" and "text" ... (string)
        - text_to_draw : a string, it could be any text you want or word.
        - text_font_size : integer to control the font size of the text 
        
        It is worth mentioning that the font style chosen is coop_black.
        It exists in the some folders of the system (the files needed are *.ttf)
        
        '''
        data = np.zeros( (np.shape(self.x)[0], np.shape(self.y)[0]) )
        
        if x_r == None:
            x_r = np.shape(self.x)[0]/2
        if y_r == None:
            y_r = np.shape(self.x)[0]/2
        if text_to_draw == None:
            text_to_draw = 'Hello World'
            
        if poly_bound_circle[0] == None:
            poly_bound_circle[0] = np.shape(self.x)[0]/2
        if poly_bound_circle[1] == None:
            poly_bound_circle[1] = np.shape(self.y)[0]/2
            
        img = Image.fromarray(data,'1')
        
        if geometry == 'circ_aperture':
            draw = ImageDraw.Draw(img)
            draw.ellipse( [ (x_r - int(a/2), y_r - int(b/2)), (x_r + int(a/2), y_r + int(b/2)) ],\
                         outline='white', fill='white' )
            aperture_diameter = np.min(( a, b ))

        if geometry == 'rect_aperture':
            draw = ImageDraw.Draw(img)
            draw.rectangle([ (x_r - int(a/2), y_r - int(b/2)), (x_r + int(a/2), y_r + int(b/2)) ],\
                           outline='white', fill='white')
            aperture_diameter = np.min(( a, b ))
            
        if geometry == 'regular_polygon':
            draw = ImageDraw.Draw(img)
            draw.regular_polygon(bounding_circle=poly_bound_circle, n_sides=poly_n_side,\
                                 rotation=poly_rotation, outline='white', fill='white')
            aperture_diameter = np.min(( poly_bound_circle[2] ))
    
        if geometry == 'text_aperture':
            font = ImageFont.truetype(coop_black, size=text_font_size)
            draw = ImageDraw.Draw(img)
            draw.multiline_text([x_r-130 , y_r-50], text=text_to_draw, fill='white', font=font)
            apaperture_diameter = text_font_size
        
        converted_data = np.array( img, dtype='int')
        # aperture_radius = aperture_radius

        return converted_data, aperture_diameter


    def wave_source(self):
        '''
        The following function is used to generate the wavefunction (spatial distibution)
        of the optical beam. The wavefunction could take many forms a plane wave, Gaussian
        super Gaussian distribution, Gauss-Laguere, Zernike, Bessel
        
        
        To be Contd.
        
        '''
        pass

    
    def angular_spectrum_frsnl(self, aperture_field, z=None):
        '''
        This function calculates the fresnel integral of the field at a certain distance z
        The calculation is done as follows. 
        
        first, we Fourier transform the field at the position z0 (aperture).
        thereafter, we calculate Fresnel integral, by multiplying Fresnel kernel 
        (Fresnel propagator) by the FT(aperture_field).
        Finally, we applied Inverse Fourier transform to recover the field at the position z.
        
        Input:
            aperture_field : the field at the aperture (z0=0)
            z : distance of propagation
        
        Output:
            diff_pattern : the diffraction pattern.
        '''

        if z == None:
            z = self.z
        
        ft_aperture_field = self.ft(aperture_field)                                         # Calculate the FT of the aperture field
        fresnel_kernel = np.exp(1j*self.k0 * z) * ((1j*self.lbd*z)**-1)\
                        * np.exp(1j * z * (self.kX**2/self.k0  + self.kY**2/self.k0 ) )                 # Define Fresnel propagator
        diff_pattern = self.ift(np.multiply( ft_aperture_field, fresnel_kernel ) )          # Inverse fourier transform
        return diff_pattern
    
    def angular_spectrum_frhffer(self, aperture_field, z=None):
        '''
        This function calculates the Fraunhoffer integral of the field at a certain distance z
        The calculation is done as follows. 
        
        first, we Fourier transform the field at the position z0 (aperture).
        thereafter, we calculate Fresnel integral, by multiplying Fresnel kernel 
        (Fresnel propagator) by the FT(aperture_field).
        Finally, we return the field calculated at the position z.
        
        Input:
            aperture_field : the field at the aperture (z0=0)
            z : distance of propagation
        
        Output:
            diff_pattern : the diffraction pattern.
        '''
        
        if z == None:
            z = self.z
        
        ft_aperture_field = self.ft(aperture_field)
        fraunhoffer_kernel = np.exp(1j*self.k0 * z) * ((1j*self.lbd*z)**-1)\
                            * np.exp(1j * z * (self.kX**2/self.k0 + self.kY**2/self.k0 ) )
        
        diff_pattern = np.multiply( ft_aperture_field, fraunhoffer_kernel ) 
        return diff_pattern
    
    
    def direct_integration_rayleigh_sommerfield_with_FT(self, aperture_field=None, z=None):
        '''
        TODO
        
        This function is not yet finished. I need to implement convolution without using the FT.
        Moreover there is a gratings effect created on the fourier transform of the RS-kernel.
        I doubt it comes from the definition of X and Y vectors.
        
        '''
        
        if z == None:
            z = self.z
        
        field_evolution = np.zeros( ( self.Mx, self.My ) )
        if aperture_field.any() == None:
            aperture_field = self.aperture_source()[0]

        
        r = np.sqrt(self.X**2 + self.Y**2 + self.z**2)

        RS_kernel = np.pad( (1/2*np.pi) * (z/r) * ( (1/r)**2 - 1j*self.k0/r)  * np.exp(1j*self.k0*r),0) 
        
        diff_pattern = fftshift( self.ift(     fftshift(self.ft(RS_kernel))*   self.ft(aperture_field)     )  )
        
        return diff_pattern
    
    
    def direct_integration_rayleigh_sommerfield(self, aperture_field=None, z=None):
        # '''
        # TODO
        
        # This function is not yet finished. I need to implement convolution without using the FFT
        
        # '''
        
        # if z == None:
        #     z = self.z

        # if aperture_field.any() == None:
        #     aperture_field = self.aperture_source()[0]
            
        # diff_pattern = np.zeros((self.Mx, self.My))
        # r = np.sqrt(self.X**2 + self.Y**2 + self.z**2)

        # RS_kernel = (1/2*np.pi) * (z/r) * ( (1/r)**2 - 1j*self.k0/r )  * np.exp(1j*self.k0*r)
        
        # # diff_pattern = convolve(aperture_field[512,:], RS_kernel[512,:])
        
        # for i in range(aperture_field.shape[0]):
        #     diff_pattern[i,:] =  convolve(RS_kernel[:,i], aperture_field[i,:], mode='same')
        #     # diff_pattern[i,:] = convolve(aperture_field[i,:], RS_kernel[i,:])
            
        
        # # for ix in range(0,self.Mx):
        # #     for iy in range(0,self.My):
        # #         ri = np.sqrt(ix**2 + iy**2 + self.z**2)
        # #         # print(aperture_field[ix,iy])
        # #         diff_pattern[ix, iy] = np.sum( np.sum(  aperture_field*\
        # #                                 (2*np.pi)**(-1) *np.exp(1j*self.k0*ri) *((ri)**-2) \
        # #                                 * self.z * (ri**(-1) - 1j*self.k0)  ) )
        # return diff_pattern
        pass
        
    
    
    def angular_spectrum_rayleigh_sommerfield(self, aperture_field, z=None):
        '''
        This function calculates the diffraction of the field at a certain distance z
        The calculation is done using the Rayleigh-Sommerfeld formula and the Angular 
        spectrum method. for more details
        please refer to the article below:
        
        https://www.osapublishing.org/ao/abstract.cfm?uri=ao-45-6-1102
        
        first, we Fourier transform the field at the position z0 (aperture).
        thereafter, we calculate Rayleigh-Sommerfeld diffraction integral
        (RS propagator) by the FT(aperture_field).
        Finally, We calculate the inverse Fourier transform.
        
        Input:
            aperture_field : the field at the aperture (z0=0)
            z : distance of propagation
        
        Output:
            diff_pattern : the diffraction pattern.
            
        N.B. For the Rayleigh diffraction formula, there is no need to specify
        two expression for the field (Fresnel, Fraunhoffer)
        
        '''
        
        if z == None:
            z = self.z
        
        ft_aperture_field = self.ft(aperture_field)
        RS_kernel =  ((1j*self.lbd*z)**-1)*np.exp( 1j*np.sqrt(  self.k0**2 - self.kX**2 - self.kY**2)*self.z )
        diff_pattern = self.ift( np.multiply(ft_aperture_field, RS_kernel) )
        return diff_pattern
    

    def Kirchoff_diffraction(self, aperture_field, z=None):
        '''
        This function implements the calculation of light diffraction using Kirchoff 
        diffraction forumula
        
        '''
        pass
    
        

if __name__ == "__main__":
    solver = DiffractionSolver(p=10, lbd=650*nm, pixel_size_x=40*um, pixel_size_y=40*um, z=100*cm, prop_method='RS')
    field_at_aperture, aperture_size = solver.aperture_source(a=60, b=60, geometry='circ_aperture', poly_rotation=55,\
                                               poly_n_side=3, text_to_draw='A Test text', text_font_size=50 , poly_bound_circle=[None, None, 100] )
    
    
    ### Using the angular spectrum method on the Fresnel Fraunhoffer expressions
    if solver.prop_method == 'frsnl_frnhfer':
        if (aperture_size*solver.dx**2)*(solver.lbd*solver.z)**-1 >= 1:    
            diff_pattern = solver.angular_spectrum_frsnl(aperture_field=field_at_aperture)
        else:
            diff_pattern = solver.angular_spectrum_frsnl(aperture_field=field_at_aperture)
        print((aperture_size*solver.dx**2)*(solver.lbd*solver.z)**-1, aperture_size*solver.dx )
    
    ### Using the angular spectrum method on the Rayleigh-Sommerfeld expressions
    elif solver.prop_method == 'RS':
        diff_pattern = solver.angular_spectrum_rayleigh_sommerfield(aperture_field=field_at_aperture)
        print((aperture_size*solver.dx**2)*(solver.lbd*solver.z)**-1, aperture_size*solver.dx )
            
    ### Using the Direct Integration method on the Rayleigh-Sommerfeld expression calculated through the Fourier Transform
    elif solver.prop_method == 'DI_FT':
        diff_pattern = solver.direct_integration_rayleigh_sommerfield_with_FT(aperture_field=field_at_aperture)
        print((aperture_size*solver.dx**2)*(solver.lbd*solver.z)**-1, aperture_size*solver.dx)
    
    # ### Using the Direct Integration method on the Rayleigh-Sommerfeld expression calculated through convolution
    # elif solver.prop_method == 'DI':
    #     diff_pattern = solver.direct_integration_rayleigh_sommerfield(aperture_field=field_at_aperture)
    #     print((aperture_size*solver.dx**2)*(solver.lbd*solver.z)**-1, aperture_size*solver.dx)

    plt.imshow(solver.intensity(diff_pattern))
    plt.show()
