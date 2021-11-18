from copy import deepcopy
import numpy as np
from math import factorial as fact
import scipy

def cart2spherical(x, y, z):
    """Converts Cartesian to spherical coordinates defined as
     - r (radius)
     - phi (longitude, in [0, 2pi]) 
     - theta (colatitude in [0, pi]) 

    Args:
        x (float): x Cartesian coordinate
        y (float): y Cartesian coordinate
        z (float): z Cartesian coordinate

    Returns:
        tuple: the spherical coordinates r, theta, phi
    """
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    if phi < 0:
        phi = phi+2*np.pi
    return r, theta, phi


def spherical2cart(r, theta, phi):
    """Converts spherical to Cartesian coordinates, inverting the
    cart2spherical function.

    Args:
        r (float): radius
        theta (float): colatitude
        phi (float): longitude

    Returns:
        tuple: the Cartesian coordinates x,y,z
    """
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z


def _single_mascon_contribution(mascon_point, mascon_mass, R0, M, l, m):
    x,y,z = mascon_point
    r, theta, phi = cart2spherical(x,y,z)
    stokesC = mascon_mass
    stokesC *= scipy.special.lpmn(m, l, np.cos(theta))[0]
    stokesS = deepcopy(stokesC)
    
    for mm in range(m+1):
        for ll in range(l+1):
            if ll < mm:
                continue
            if mm==0:
                delta=1
            else:
                delta=0   
            coeff1 = (r/R0)**ll
            coeff2C = np.cos(mm*phi)
            coeff2S = np.sin(mm*phi)
            coeff3 = (2-delta)*np.math.factorial(ll-mm)/np.math.factorial(ll+mm)
            normalized = np.sqrt(np.math.factorial(ll+mm) / (2-delta)/(2*ll+1)/np.math.factorial(ll-mm))
            stokesS[mm,ll] *= coeff1*coeff2S*coeff3*normalized
            stokesC[mm,ll] *= coeff1*coeff2C*coeff3*normalized 
    return (stokesC / M, stokesS / M)


def mascon2stokes(mascon_points, mascon_masses, R0, M, l, m):
    """Computes the stokes coefficients out of a mascon model

    Args:
        mascon_points (array(N,3)): cartesian positions of the mascon points
        mascon_masses (array(N,)): masses of the mascons
        R0 (float): characteristic radius (oftne the mean equatorial) of the body
        M (float): total mass of the body
        l (int): degree
        m (int): order (must be less or equal the degree)

    Returns:
        array(l+1,m+1), array(l+1,m+1): Stokes coefficients (Clm, Slm)
    """
    stokesS = 0 
    stokesC = 0 
    for point, mass in zip(mascon_points,mascon_masses):
        tmpC, tmpS = _single_mascon_contribution(point, mass, R0, M, l, m)
        stokesS+=tmpS
        stokesC+=tmpC
    return (stokesC, stokesS)