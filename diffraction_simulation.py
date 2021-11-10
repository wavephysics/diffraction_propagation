# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 09:57:30 2021

@author: OmarBoughdad
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.fftpack import fft2, ifft2, fftshift, fftfreq
from scipy.special import comb, factorial
from PIL import Image, ImageDraw, ImageFont


arial = 'arial.ttf'
coop_black = 'COOPBL.TTF'


### DEFINE PHYSICAL AND FOURIER SPACES

mm = 1e-3
cm = mm*1e1
um = mm*mm
nm = um*mm
lbd = 650*nm
k0 = 2*np.pi/lbd
p = 11  # This is a integer to define the number of point
Lx = 120*mm  # Physical length along x
Ly = 120*mm  # Physical length along y
nx = 2**p
ny = 2**p
M = nx + 1

xn = np.linspace(-int(M/2) , int(M/2), M)
yn = np.linspace(-int(M/2) , int(M/2), M)

dx = Lx/nx   # Grid size (pixel size) along x
dy = Ly/ny   # Grid size (pixel size) along y
x = xn*dx
y = yn*dy

dkx = 2*np.pi / Lx
dky = 2*np.pi / Ly

X,Y = np.meshgrid(x,y)

# kx = np.linspace(-(dkx / 2 ) * M, dkx / 2 * M, M)
# ky = np.linspace(-(dky / 2 ) * M, dky / 2 * M, M)
kx = np.fft.fftfreq(M, d=dx*1e6)
ky = np.fft.fftfreq(M, d=dy*1e6)
kx.sort(); ky.sort()
kX, kY = np.meshgrid(kx, ky)

aperture_radius = 300    # In pixels

def intensity(field):
    return np.real(field*np.conj(field))

def ft(field):
    return fftshift(fft2(field))
    
def ift(field):
    return ifft2(field)

def shft(x):
    fftshift(x)

def aperture(a=aperture_radius, b=aperture_radius, poly_bound_circle=(np.shape(x)[0]/2, np.shape(x)[0]/2, 200), poly_n_side=3, poly_rotation=0,\
             x_r=None, y_r=None, geometry='circ_aperture', text_to_draw=None, text_font_size=50):
    data = np.zeros( (np.shape(x)[0], np.shape(y)[0]) )
    x_r = np.shape(x)[0]/2
    y_r = np.shape(x)[0]/2
    img = Image.fromarray(data,'1')

    if geometry == 'circ_aperture':
        draw = ImageDraw.Draw(img)
        draw.ellipse( [ (x_r - int(a/2), y_r - int(b/2)), (x_r + int(a/2), y_r + int(b/2)) ], outline='white', fill='white' )
        
    if geometry == 'rect_aperture':
        draw = ImageDraw.Draw(img)
        draw.rectangle([ (x_r - int(a/2), y_r - int(b/2)), (x_r + int(a/2), y_r + int(b/2)) ], outline='white', fill='white')
        
    if geometry == 'regular_polygon':
        draw = ImageDraw.Draw(img)
        draw.regular_polygon(bounding_circle=poly_bound_circle, n_sides=poly_n_side, rotation=poly_rotation, outline='white', fill='white')
    
    if geometry == 'text':
        font = ImageFont.truetype(coop_black, size=text_font_size)
        draw = ImageDraw.Draw(img)
        draw.multiline_text([ x_r-200 , y_r-50], text=text_to_draw, fill='white', font=font)
    
    converted_data = np.array( img, dtype='int')
    return converted_data

def frsnl_prop(z=50*cm, field_z0=None ):
    pass
    
def direct_int():
    pass

def angular_spec():
    pass



    
A0 = 1
src = A0*aperture(poly_n_side=6, geometry='circ_aperture', text_to_draw='Envisics', text_font_size=100)

z = 10*cm   # Distance of propagation

frsnl_number = (aperture_radius*dx)**2/(lbd*z)

# h1 = np.exp( 1j*k0/(2*z)* (X**2 + Y**2) )

# h1 = np.exp(1j*k0*z)*((1j*lbd*z)**-1 ) * np.exp( 1j*k0/(2*z) *( kX**2  + kY**2))
ft_src = ft( src ) 

H = np.exp(1j*k0 * z) * ((1j*lbd*z)**-1) * np.exp(1j *k0 * z * (X**2+ Y**2) )

result = ift(np.multiply( ft_src, H ) )

# hxy = np.exp(1j*k0*z)* ((1j*lbd*z)**-1 ) * np.exp( 1j*k0/(2*z) *( X**2  + Y**2))


# TODO 






