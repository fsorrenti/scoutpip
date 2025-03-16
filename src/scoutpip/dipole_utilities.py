import numpy as np
from astropy.coordinates import SkyCoord, spherical_to_cartesian
from astropy import units as u
import math as m
from scipy import integrate, special
from astropy import coordinates


"""
Firstly all the geometrical/positional stuff
"""
# let's move from equatorial to galactic coordinates

# https://astronomy.stackexchange.com/questions/39404/how-to-plot-celestial-equator-in-galactic-coordinates-why-does-my-plot-appear


def map_angle_1pi(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [0, pi]
    """
    return -angle + np.pi / 2


def map_angle_2pi(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [0, 2pi]
    """
    return (angle + 2 * np.pi) % (2 * np.pi)


def normalize_angle_180(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [-90, 90]
    """
    while angle > 90:
        angle -= 90

    while angle < -90:
        angle += 90

    return angle


def eq2gal(ra, dec):
    """
    Transforms equatorial coordinates to galactic ones.
    Then prepares them for matplotlib aitoff projection.
    """

    eq = SkyCoord(dec=dec * u.degree, ra=ra * u.degree, frame="icrs")
    gal = eq.galactic

    l_gal, b_gal = gal.l.wrap_at("180d").radian, gal.b.radian

    return l_gal, b_gal


def sph2cart(r, l, b):
    rsin_theta = r * np.sin(b)
    x = rsin_theta * np.cos(l)
    y = rsin_theta * np.sin(l)
    z = r * np.cos(b)
    return [x, y, z]


def difference_vector(v1, ra1, dec1, v2, ra2, dec2):
    # for spherical to cartesian firstly lat(dec) and then long(ra)
    vector1 = np.array(
        coordinates.spherical_to_cartesian(v1, np.deg2rad(dec1), np.deg2rad(ra1))
    )
    vector2 = np.array(
        coordinates.spherical_to_cartesian(v2, np.deg2rad(dec2), np.deg2rad(ra2))
    )
    difference = vector1 - vector2
    # difference_polar=np.array(coordinates.cartesian_to_spherical(difference[0],difference[1], difference[2]))

    # I put the return in order to obtain firstly the ra and the the dec
    return (
        coordinates.cartesian_to_spherical(difference[0], difference[1], difference[2])[
            0
        ],
        coordinates.cartesian_to_spherical(difference[0], difference[1], difference[2])[
            2
        ].degree,
        coordinates.cartesian_to_spherical(difference[0], difference[1], difference[2])[
            1
        ].degree,
    )


def difference_vector_by_hand(v1, ra1, dec1, v2, ra2, dec2):
    ra1 = np.deg2rad(ra1)
    ra2 = np.deg2rad(ra2)
    dec1 = np.deg2rad(dec1)
    dec2 = np.deg2rad(dec2)

    x = v1 * np.cos(dec1) * np.cos(ra1) - v2 * np.cos(dec2) * np.cos(ra2)
    y = v1 * np.cos(dec1) * np.sin(ra1) - v2 * np.cos(dec2) * np.sin(ra2)
    z = v1 * np.sin(dec1) - v2 * np.sin(dec2)

    v = np.sqrt(x**2 + y**2 + z**2)
    ra = np.rad2deg(np.arctan(y / x)) + 180
    dec = np.rad2deg(np.arcsin(z / v))
    return v, ra, dec


#import uncertainties.umath as um
#from uncertainties import ufloat

'''
def difference_vector_by_hand_uncertainties(v1, errorv1, ra1, errorra1, dec1, errordec1, v2, errorv2, ra2, errorra2, dec2, errordec2,):
    v1_with_error=ufloat(v1, errorv1)
    ra1_with_error=ufloat(ra1, errorra1)
    dec1_with_error=ufloat(dec1, errordec1)
    v2_with_error=ufloat(v2, errorv2)
    ra2_with_error=ufloat(ra2, errorra2)
    dec2_with_error=ufloat(dec2, errordec2)


    ra1_with_error_rad = um.radians(ra1_with_error)
    ra2_with_error_rad = um.radians(ra2_with_error)
    dec1_with_error_rad = um.radians(dec1_with_error)
    dec2_with_error_rad = um.radians(dec2_with_error)

    x = v1_with_error * um.cos(dec1_with_error_rad) * um.cos(ra1_with_error_rad) - v2_with_error * um.cos(dec2_with_error_rad) * um.cos(ra2_with_error_rad)
    y = v1_with_error * um.cos(dec1_with_error_rad) * um.sin(ra1_with_error_rad) - v2_with_error * um.cos(dec2_with_error_rad) * um.sin(ra2_with_error_rad)
    z = v1_with_error * um.sin(dec1_with_error_rad) - v2_with_error * um.sin(dec2_with_error_rad)

    v = um.sqrt(x**2 + y**2 + z**2)
    ra = um.degrees(um.atan2(y , x)) + 180
    dec = um.degrees(um.asin(z / v))
    return v, ra, dec
'''

"""filter dataset functions"""


def chisq_filtered(
    dataset,
    inversed_covariance,
    horstmann,
    lower_bound,
    v,
    H0,
    omega_m,
    ra,
    dec,
    M_correction=None,
    M=None,
):
    """this function will give us the chisquare for the filtered dataset in our dataset, in Planck and with fixed position"""
    z_prefilter = dataset[:, 0]
    is_cal_prefilter = dataset[:, 5]
    filtered_data_file = filter_z_cepheid(
        dataset, inversed_covariance, z_prefilter, is_cal_prefilter, lower_bound, 6
    )[0]
    filtered_inversed_covariance = filter_z_cepheid(
        dataset, inversed_covariance, z_prefilter, is_cal_prefilter, lower_bound, 6
    )[1]

    z_filtered = filtered_data_file[:, 0]
    ra_filtered = filtered_data_file[:, 6]
    dec_filtered = filtered_data_file[:, 7]
    mu_filtered = filtered_data_file[:, 2]
    ceph_dis_filtered = filtered_data_file[:, 4]
    is_cal_filtered = filtered_data_file[:, 5]

    if M_correction:
        residual_our_analysis = exp_obs_dipole_with_M(
            v,
            M,
            H0,
            omega_m,
            ra,
            dec,
            ra_filtered,
            dec_filtered,
            mu_filtered,
            z_filtered,
            horstmann,
            is_cal_filtered,
            ceph_dis_filtered,
            False,
        )
    else:
        residual_our_analysis = exp_obs_dipole(
            v,
            H0,
            omega_m,
            ra,
            dec,
            ra_filtered,
            dec_filtered,
            mu_filtered,
            z_filtered,
            horstmann,
            is_cal_filtered,
            ceph_dis_filtered,
            False,
        )

    return chisq(residual_our_analysis, filtered_inversed_covariance)


def chisq_filtered_monopole(
    dataset,
    inversed_covariance,
    horstmann,
    lower_bound,
    H0,
    omega_m,
    z_hd=None,
    M_correction=None,
    M=None,
):
    """this function will give us the chisquare FOR THE MONOPOLE HYPOTHESIS for the filtered dataset in our dataset, in Planck and with fixed position"""

    z_prefilter = dataset[:, 0]
    is_cal_prefilter = dataset[:, 5]

    filtered_data_file = filter_z_cepheid(
        dataset, inversed_covariance, z_prefilter, is_cal_prefilter, lower_bound, 6
    )[0]
    filtered_inversed_covariance = filter_z_cepheid(
        dataset, inversed_covariance, z_prefilter, is_cal_prefilter, lower_bound, 6
    )[1]

    z_filtered = filtered_data_file[:, 0]
    z_hd_filtered = filtered_data_file[:, 9]
    mu_filtered = filtered_data_file[:, 2]
    ceph_dis_filtered = filtered_data_file[:, 4]
    is_cal_filtered = filtered_data_file[:, 5]

    if M_correction:
        if z_hd:  # in this way I use z_hd instead of z_hel
            residual_our_analysis = exp_obs_monopole_with_M(
                M,
                H0,
                omega_m,
                mu_filtered,
                z_hd_filtered,
                horstmann,
                is_cal_filtered,
                ceph_dis_filtered,
                False,
            )
        else:
            residual_our_analysis = exp_obs_monopole_with_M(
                M,
                H0,
                omega_m,
                mu_filtered,
                z_filtered,
                horstmann,
                is_cal_filtered,
                ceph_dis_filtered,
                False,
            )
    else:
        if z_hd:  # in this way I use z_hd instead of z_hel
            residual_our_analysis = exp_obs_monopole(
                H0,
                omega_m,
                mu_filtered,
                z_hd_filtered,
                horstmann,
                is_cal_filtered,
                ceph_dis_filtered,
                False,
            )
        else:
            residual_our_analysis = exp_obs_monopole(
                H0,
                omega_m,
                mu_filtered,
                z_filtered,
                horstmann,
                is_cal_filtered,
                ceph_dis_filtered,
                False,
            )

    return chisq(residual_our_analysis, filtered_inversed_covariance)


def symmetric_traceless_matrix(a11, a12, a13, a22, a23):
    """How to construct the symmetric traceless matrix from the individual matrix elements"""
    return np.array([[a11, a12, a13], [a12, a22, a23], [a13, a23, -(a11 + a22)]])


def factor_scalar(ra_sun_in_deg, dec_sun_in_deg, ra, dec):
    # for spherical-cartesian first latitude [-pi/2,pi/2] and then longitude [0, 2pi]
    # I move to cartesian coordinate

    sun_cartesian = spherical_to_cartesian(
        1, np.deg2rad(dec_sun_in_deg), np.deg2rad(ra_sun_in_deg)
    )

    return np.dot(
        sun_cartesian,
        spherical_to_cartesian(
            1,
            np.deg2rad(dec),
            np.deg2rad(ra),
        ),
    )


def factor_scalar_matrix(matrix, ra_1_in_deg, dec_1_in_deg, ra_2_in_deg, dec_2_in_deg):
    # for spherical-cartesian first latitude [-pi/2,pi/2] and then longitude [0, 2pi]
    # I move to cartesian coordinate

    cartesian_1 = spherical_to_cartesian(
        1, np.deg2rad(dec_1_in_deg), np.deg2rad(ra_1_in_deg)
    )

    cartesian_2 = spherical_to_cartesian(
        1, np.deg2rad(dec_2_in_deg), np.deg2rad(ra_2_in_deg)
    )

    return cartesian_1 @ matrix @ cartesian_2


# convert the column systematic into a matrix
def conversion_no_statistical(name_file, dimension):
    input = np.genfromtxt(name_file)  # niente virgoletta dato il duck typing
    matrix = np.zeros([dimension, dimension])
    for j in range(0, dimension - 1):
        for i in range(0, dimension - 1):
            matrix[i, j] = input[dimension * j + i]
    return matrix


def covariance_pantheon(sist_matrix, statistical_error):
    l = len(statistical_error)
    statistical_matrix = np.zeros([l, l])
    for i in range(0, l):
        statistical_matrix[i, i] = statistical_error[i] * statistical_error[i]

    return sist_matrix + statistical_matrix


def chisq(x, inv_cov):
    xt = np.transpose(x)
    return np.dot(np.dot(xt, inv_cov), x)


def integrand(z, H0, Omat):
    return 1 / (H0 * m.sqrt(Omat * (1 + z) ** 3 + (1 - Omat)))

"""
Let's put all thr agnostic part
"""

def coeff_expansion(coefficient, ra, dec):
    if isinstance(coefficient, (int, float)):
        return np.full(len(ra), coefficient)
    else:        
        coefficient_vector=np.array(coefficient[1:4])
        product=np.dot(
        coefficient_vector,
        spherical_to_cartesian(
            1,
            np.deg2rad(dec),
            np.deg2rad(ra),
        ),
    )
        return coefficient[0]+product
    
def dl_agnostic(z, H_inverse_agnostic, q, j,ra, dec):
    #c = 299792.458
    coeff1=coeff_expansion( H_inverse_agnostic, ra, dec)
    coeff2=coeff_expansion( q, ra, dec)
    coeff3=coeff_expansion( j,ra, dec)

    term1=coeff1
    term2=coeff1*(1-coeff2)/2
    term3=coeff1*(-1+3*coeff2**2+coeff2-coeff3)/6

    return z*term1+z**2*term2+z**3*term3



def exp_obs_agnostic(
    H_inverse_agnostic, q, j, dM, mu_function, z_function, ra_function, dec_function, horstmann, is_cal_function=None, ceph_function=None, no_cepheid=None
):
    # Calculate luminosity distances
    dl_values = dl_agnostic(z_function, H_inverse_agnostic, q, j, ra_function, dec_function)
    
    # Initialize array for expected observations
    l=len(mu_function)
    exp_obs_agn = np.empty(l)

    # Check conditions for Horstmann or no cepheid cases
    if horstmann or no_cepheid:
        # Calculate expected observations without calibration correction
        exp_obs_agn = mu_function - 5 * np.log10(dl_values) - 25
    else:
        # Mask for elements needing calibration correction
        cal_mask = (is_cal_function == 1)
        
        # Apply calibration correction for masked elements
        exp_obs_agn[cal_mask] = mu_function[cal_mask] + dM - ceph_function[cal_mask]
        
        # Apply calibration correction for unmasked elements
        #the tilde means "False"
        exp_obs_agn[~cal_mask] = mu_function[~cal_mask] + dM - 5 * np.log10(dl_values[~cal_mask]) - 25

    return exp_obs_agn


def exp_obs_agnostic_redshift_dipole(
    H_inverse_agnostic, q, j, v0, ra0, dec0, dM, mu_function, z_function, ra_function, dec_function, horstmann, is_cal_function=None, ceph_function=None, no_cepheid=None
):
    #calculate corrected redshift with free dipole
    z_corrected_free_dipole=z_cmb_cross(z_function, v0, ra0, dec0, ra_function, dec_function)


    # Calculate luminosity distances
    dl_values = dl_agnostic(z_corrected_free_dipole, H_inverse_agnostic, q, j, ra_function, dec_function)
    
    # Initialize array for expected observations
    l=len(mu_function)
    exp_obs_agn = np.empty(l)

    # Check conditions for Horstmann or no cepheid cases
    if horstmann or no_cepheid:
        # Calculate expected observations without calibration correction
        exp_obs_agn = mu_function - 5 * np.log10(dl_values) - 25
    else:
        # Mask for elements needing calibration correction
        cal_mask = (is_cal_function == 1)
        
        # Apply calibration correction for masked elements
        exp_obs_agn[cal_mask] = mu_function[cal_mask] + dM - ceph_function[cal_mask]
        
        # Apply calibration correction for unmasked elements
        #the tilde means "False"
        exp_obs_agn[~cal_mask] = mu_function[~cal_mask] + dM - 5 * np.log10(dl_values[~cal_mask]) - 25

    return exp_obs_agn


def dl_agnostic_taylor_expansion(z, d1, d2, d3):
    c = 299792.458
    #we multiply by c in order to simplify our results and prior
    return c*(z*d1+z**2*d2/2+z**3*d3/6)

def dl_agnostic_taylor_expansion_monopole(z, d0, d1, d2, d3):
    c = 299792.458
    #we multiply by c in order to simplify our results and prior
    return c*(d0+z*d1+z**2*d2/2+z**3*d3/6)



def exp_obs_agnostic_taylor_expansion(
     d1, d2, d3, dM, mu_function, z_function, horstmann, is_cal_function=None, ceph_function=None, no_cepheid=None
):
    # Calculate luminosity distances
    dl_values = dl_agnostic_taylor_expansion(z_function, d1, d2, d3)
    
    # Initialize array for expected observations
    l=len(mu_function)
    exp_obs_agn = np.empty(l)

    # Check conditions for Horstmann or no cepheid cases
    if horstmann or no_cepheid:
        # Calculate expected observations without calibration correction
        exp_obs_agn = mu_function - 5 * np.log10(dl_values) - 25
    else:
        # Mask for elements needing calibration correction
        cal_mask = (is_cal_function == 1)
        
        # Apply calibration correction for masked elements
        exp_obs_agn[cal_mask] = mu_function[cal_mask] + dM - ceph_function[cal_mask]
        
        # Apply calibration correction for unmasked elements
        #the tilde means "False"
        exp_obs_agn[~cal_mask] = mu_function[~cal_mask] + dM - 5 * np.log10(dl_values[~cal_mask]) - 25

    return exp_obs_agn

def exp_obs_agnostic_taylor_expansion_redshift_dipole_monopole(
     d0, d1, d2, d3,  v0, ra0, dec0, dM, mu_function, z_function, ra_function, dec_function, horstmann,  is_cal_function=None, ceph_function=None, no_cepheid=None
):

    #calculate corrected redshift with free dipole
    z_corrected_free_dipole=z_cmb_cross(z_function, v0, ra0, dec0, ra_function, dec_function)

    # Calculate luminosity distances
    dl_values = dl_agnostic_taylor_expansion_monopole(z_corrected_free_dipole, d0, d1, d2, d3)
    
    # Initialize array for expected observations
    l=len(mu_function)
    exp_obs_agn = np.empty(l)

    # Check conditions for Horstmann or no cepheid cases
    if horstmann or no_cepheid:
        # Calculate expected observations without calibration correction
        exp_obs_agn = mu_function - 5 * np.log10(dl_values) - 25
    else:
        # Mask for elements needing calibration correction
        cal_mask = (is_cal_function == 1)
        
        # Apply calibration correction for masked elements
        exp_obs_agn[cal_mask] = mu_function[cal_mask] + dM - ceph_function[cal_mask]
        
        # Apply calibration correction for unmasked elements
        #the tilde means "False"
        exp_obs_agn[~cal_mask] = mu_function[~cal_mask] + dM - 5 * np.log10(dl_values[~cal_mask]) - 25

    return exp_obs_agn



def exp_obs_agnostic_taylor_expansion_redshift_dipole_monopole_inserted(
     z0, d1, d2, d3,  v0, ra0, dec0, dM, mu_function, z_function, ra_function, dec_function, horstmann,  is_cal_function=None, ceph_function=None, no_cepheid=None
):
    """
    We add also the monopole in the redshift
    """
    #calculate corrected redshift with free dipole
    z_corrected_free_monopole_dipole=z_monopole_dipole_cross(z_function, z0, v0, ra0, dec0, ra_function, dec_function)

    # Calculate luminosity distances
    dl_values = dl_agnostic_taylor_expansion(z_corrected_free_monopole_dipole, d1, d2, d3)
    
    # Initialize array for expected observations
    l=len(mu_function)
    exp_obs_agn = np.empty(l)

    # Check conditions for Horstmann or no cepheid cases
    if horstmann or no_cepheid:
        # Calculate expected observations without calibration correction
        exp_obs_agn = mu_function - 5 * np.log10(dl_values) - 25
    else:
        # Mask for elements needing calibration correction
        cal_mask = (is_cal_function == 1)
        
        # Apply calibration correction for masked elements
        exp_obs_agn[cal_mask] = mu_function[cal_mask] + dM - ceph_function[cal_mask]
        
        # Apply calibration correction for unmasked elements
        #the tilde means "False"
        exp_obs_agn[~cal_mask] = mu_function[~cal_mask] + dM - 5 * np.log10(dl_values[~cal_mask]) - 25

    return exp_obs_agn


def exp_obs_agnostic_taylor_expansion_redshift_dipole_quadrupole_monopole_inserted(
     z0, d1, d2, d3, v0, ra0, dec0,a11,a12,a13,a22,a23, 
     dM, mu_function, z_function, ra_function, dec_function, horstmann,  
     is_cal_function=None, ceph_function=None, no_cepheid=None
):
    """
    We add also the monopole in the redshift
    """
    #calculate corrected redshift with free dipole
    
    c = 0
    alpha_matrix = symmetric_traceless_matrix(a11, a12, a13, a22, a23)
    number_of_elements = len(mu_function)
    exp_obs_mon = np.zeros(number_of_elements)
    if horstmann:
        for i in range(0, number_of_elements):
            exp_obs_mon[i] = mu_function[i] - 5 * np.log10(
                dl_agnostic_taylor_expansion(z_agnostic_monopole_dipole_quadrupole_cross(z_function[i],
                                                    z0,
                                                    v0,
                                                    ra0,
                                                    dec0,
                                                    alpha_matrix, 
                                                    ra_function[i],
                                                    dec_function[i],       
                                                    ), d1, d2, d3))- 25
    else:
        if no_cepheid:
            for i in range(0, number_of_elements):
                exp_obs_mon[i] = mu_function[i] -  5 * np.log10(
                dl_agnostic_taylor_expansion(z_agnostic_monopole_dipole_quadrupole_cross(z_function[i],
                                                    z0,
                                                    v0,
                                                    ra0,
                                                    dec0,
                                                    alpha_matrix, 
                                                    ra_function[i],
                                                    dec_function[i],       
                                                    ), d1, d2, d3))- 25
        else:
            for i in range(0, number_of_elements):
                # print(z_quadrupole_corrected(v_bulk, ra_bulk, dec_bulk, H0, Omat, alpha_matrix, v_sun, z[i], ra_sun_in_deg, dec_sun_in_deg, ra, dec))

                if is_cal_function[i] == 1:
                    c = c + 1
                    exp_obs_mon[i] = mu_function[i] + dM - ceph_function[i]
                else:
                    exp_obs_mon[i] = (
                        mu_function[i]
                        + dM
                        -  5 * np.log10(
                dl_agnostic_taylor_expansion(z_agnostic_monopole_dipole_quadrupole_cross(z_function[i],
                                                    z0,
                                                    v0,
                                                    ra0,
                                                    dec0,
                                                    alpha_matrix, 
                                                    ra_function[i],
                                                    dec_function[i],       
                                                    ), d1, d2, d3))- 25
                    )
    return exp_obs_mon



def exp_obs_agnostic_taylor_expansion_redshift_dipole(
    d1, d2, d3,  v0, ra0, dec0, dM, mu_function, z_function, ra_function, dec_function, horstmann,  is_cal_function=None, ceph_function=None, no_cepheid=None
):
    
    #calculate corrected redshift with free dipole
    z_corrected_free_dipole=z_cmb_cross(z_function, v0, ra0, dec0, ra_function, dec_function)

    # Calculate luminosity distances
    dl_values = dl_agnostic_taylor_expansion(z_corrected_free_dipole, d1, d2, d3)
    
    # Initialize array for expected observations
    l=len(mu_function)
    exp_obs_agn = np.empty(l)

    # Check conditions for Horstmann or no cepheid cases
    if horstmann or no_cepheid:
        # Calculate expected observations without calibration correction
        exp_obs_agn = mu_function - 5 * np.log10(dl_values) - 25
    else:
        # Mask for elements needing calibration correction
        cal_mask = (is_cal_function == 1)
        
        # Apply calibration correction for masked elements
        exp_obs_agn[cal_mask] = mu_function[cal_mask] + dM - ceph_function[cal_mask]
        
        # Apply calibration correction for unmasked elements
        #the tilde means "False"
        exp_obs_agn[~cal_mask] = mu_function[~cal_mask] + dM - 5 * np.log10(dl_values[~cal_mask]) - 25

    return exp_obs_agn



"""
Let's analyse all the stuff for the quadrupole term
"""


def omm(z, omegam0):
    return (omegam0 * (1 + z) ** 3) / (1 + omegam0 * z * (3 + z * (3 + z)))


def Hz(z, H0, omegam0):
    return H0 * ((1 - omegam0) + omegam0 * (1 + z) ** 3) ** (1 / 2)


def D1z(z, omegam0):
    return (
        6
        * special.hyp2f1(1 / 3, 1, 11 / 6, (1 - 1 / omm(z, omegam0)))
        / (5 * omegam0 * (1 + z))
    )


def fz(z, omegam0):
    return (
        omm(z, omegam0)
        * (5 / special.hyp2f1(1 / 3, 1, 11 / 6, (1 - 1 / omm(z, omegam0))) - 3)
        / 2
    )


def DD1(z, omegam0):
    return -D1z(z, omegam0) * fz(z, omegam0) / (1 + z)


def prefactor_z(z, H0, Omat):
    return (D1z(z, Omat) * fz(z, Omat) * Hz(z, H0, Omat)) / (
        D1z(0, Omat) * fz(0, Omat) * Hz(0, H0, Omat) * (1 + z)
    )


def dl_monopole(z, H0, Omat):
    c = 299792.458
    return (
        c * (1 + z) * integrate.romberg(integrand, 0, z, args=(H0, Omat))
    )  # return quad(integrand, 0, z, args=(H0, Omat, Olam))[0]


def dl_dipole(v, z, H0, Omat, ra_sun_in_deg, dec_sun_in_deg, ra, dec, v_pec=None):
    scalar = factor_scalar(ra_sun_in_deg, dec_sun_in_deg, ra, dec)
    if v_pec is not None:  # Check if v_pec is not None
        return ((scalar * (v) - v_pec) * (1 + z) ** 2) / Hz(
            z,
            H0,
            Omat,
        )  # I use the plus for mocking
    else:
        return (scalar * (v) * (1 + z) ** 2) / Hz(z, H0, Omat)


def dl_quadrupole_with_bulk(
    v_bulk, ra_bulk, dec_bulk, H0, Omat, alpha_matrix, z, ra, dec
):
    scalar_bulk = factor_scalar(ra_bulk, dec_bulk, ra, dec)
    scalar_bulk_matrix = factor_scalar_matrix(alpha_matrix, ra, dec, ra, dec)
    return (
        prefactor_z(z, H0, Omat)
        * (scalar_bulk * v_bulk + scalar_bulk_matrix)
        * (1 + z) ** 2
    ) / Hz(z, H0, Omat)


def dl_bulk(v_bulk, ra_bulk, dec_bulk, H0, Omat, z, ra, dec):
    scalar_bulk = factor_scalar(ra_bulk, dec_bulk, ra, dec)
    return (1 + z) ** 2 * DD1(z, H0, Omat) * (scalar_bulk * v_bulk) / Hz(z, H0, Omat)


def monopole(z, H0, Omat):
    return 5 * np.log10((dl_monopole(z, H0, Omat))) + 25

def monopole_dl_extra_corrected(beta, v_bulk, ra_bulk, dec_bulk, H0, Omat, alpha_matrix,v_sun,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec,
    z):
    
    c = 299792.458
    scalar_bulk = factor_scalar(ra_bulk, dec_bulk, ra, dec)
    scalar_bulk_matrix = factor_scalar_matrix(alpha_matrix, ra, dec, ra, dec)

    z_q=z_monopole_bulk_quadrupole(
    z,
    beta,
    v_bulk,
    ra_bulk,
    dec_bulk,
    H0,
    Omat,
    alpha_matrix,
    v_sun,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec)

    return 5 * np.log10((dl_monopole(z_q, H0, Omat))-dl_monopole(z, H0, Omat)*(scalar_bulk * v_bulk + scalar_bulk_matrix)/c) + 25

def dipole(v, z, H0, Omat, ra_sun_in_deg, dec_sun_in_deg, ra, dec, v_pec=None):
    if v_pec is not None:  # Check if v_pec is not None
        return (
            5
            * np.log10(
                (
                    dl_monopole(z, H0, Omat)
                    + dl_dipole(
                        v, z, H0, Omat, ra_sun_in_deg, dec_sun_in_deg, ra, dec, v_pec
                    )
                )
            )
            + 25
        )
    else:
        return (
            5
            * np.log10(
                (
                    dl_monopole(z, H0, Omat)
                    + dl_dipole(v, z, H0, Omat, ra_sun_in_deg, dec_sun_in_deg, ra, dec)
                )
            )
            + 25
        )


def quadrupole(
    v_bulk,
    ra_bulk,
    dec_bulk,
    H0,
    Omat,
    alpha_matrix,
    v,
    z,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec,
):
    return (
        5
        * np.log10(
            (
                dl_monopole(z, H0, Omat)
                + dl_dipole(v, z, H0, Omat, ra_sun_in_deg, dec_sun_in_deg, ra, dec)
                - dl_quadrupole_with_bulk(
                    v_bulk, ra_bulk, dec_bulk, H0, Omat, alpha_matrix, z, ra, dec
                )
            )
        )
        + 25
    )


def bulk(
    v_bulk, ra_bulk, dec_bulk, H0, Omat, v, z, ra_sun_in_deg, dec_sun_in_deg, ra, dec
):
    return (
        5
        * np.log10(
            (
                dl_monopole(z, H0, Omat)
                + dl_dipole(v, z, H0, Omat, ra_sun_in_deg, dec_sun_in_deg, ra, dec)
                - dl_bulk(v_bulk, ra_bulk, dec_bulk, H0, Omat, z, ra, dec)
            )
        )
        + 25
    )


# QUAD, INTEGRAZIONE IN GENERALE HA PROBLEMI CON ARRAY: TOCCA USARE CICLI FOR
# I define the exp-obs stuff for the monopole
def exp_obs_monopole(
    H0, omega_m, mu, z, horstmann, is_cal=None, ceph=None, no_cepheid=None
):
    c = 0
    number_of_elements = len(mu)
    exp_obs_mon = np.zeros(number_of_elements)
    if horstmann:
        for i in range(0, number_of_elements):
            exp_obs_mon[i] = mu[i] - monopole(z[i], H0, omega_m)

    else:
        if no_cepheid:
            for i in range(0, number_of_elements):
                exp_obs_mon[i] = mu[i] - monopole(z[i], H0, omega_m)
        else:
            for i in range(0, number_of_elements):
                if is_cal[i] == 1:
                    c = c + 1
                    exp_obs_mon[i] = mu[i] - ceph[i]
                    # exp_obs_mon[i]=mu[i]-monopole(z6[i],H0,omega_m)
                else:
                    exp_obs_mon[i] = mu[i] - monopole(z[i], H0, omega_m)

    return exp_obs_mon


def exp_obs_monopole_with_M(
    M, H0, omega_m, mu, z, horstmann, is_cal=None, ceph=None, no_cepheid=None
):
    c = 0
    number_of_elements = len(mu)
    exp_obs_mon = np.zeros(number_of_elements)
    if horstmann:
        for i in range(0, number_of_elements):
            exp_obs_mon[i] = mu[i] - monopole(z[i], H0, omega_m)

    else:
        if no_cepheid:
            for i in range(0, number_of_elements):
                exp_obs_mon[i] = mu[i] - monopole(z[i], H0, omega_m)
        else:
            for i in range(0, number_of_elements):
                if is_cal[i] == 1:
                    c = c + 1
                    exp_obs_mon[i] = mu[i] + M - ceph[i]
                    # exp_obs_mon[i]=mu[i]-monopole(z6[i],H0,omega_m)
                else:
                    exp_obs_mon[i] = mu[i] + M - monopole(z[i], H0, omega_m)

    return exp_obs_mon


# riess et al. 2016 2.4%
def exp_obs_dipole_horstmann(
    velocity, H0, Omat, ra_sun_in_deg, dec_sun_in_deg, ra, dec, mu, z
):
    number_of_elements = len(mu)
    exp_obs_dip = np.zeros(number_of_elements)
    for i in range(0, number_of_elements):
        exp_obs_dip[i] = mu[i] - dipole(
            velocity, z[i], H0, Omat, ra_sun_in_deg, dec_sun_in_deg, ra[i], dec[i]
        )
    return exp_obs_dip


def exp_obs_dipole(
    velocity,
    M,
    H0,
    Omat,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec,
    mu,
    z,
    horstmann,
    is_cal=None,
    ceph=None,
    no_cepheid=None,
):
    number_of_elements = len(mu)
    exp_obs_dip = np.zeros(number_of_elements)

    if horstmann:
        for i in range(0, number_of_elements):
            exp_obs_dip[i] = mu[i] - dipole(
                velocity, z[i], H0, Omat, ra_sun_in_deg, dec_sun_in_deg, ra[i], dec[i]
            )
    else:
        if no_cepheid:
            for i in range(0, number_of_elements):
                exp_obs_dip[i] = mu[i] - dipole(
                    velocity,
                    z[i],
                    H0,
                    Omat,
                    ra_sun_in_deg,
                    dec_sun_in_deg,
                    ra[i],
                    dec[i],
                )
        else:
            for i in range(0, number_of_elements):
                if is_cal[i] == 1:
                    exp_obs_dip[i] = mu[i] + M - ceph[i]
                    # exp_obs_dip[i]=mu[i]-dipole(velocity, z7[i],73.4,0.338,scalar[i])
                else:
                    exp_obs_dip[i] = (
                        mu[i]
                        + M
                        - dipole(
                            velocity,
                            z[i],
                            H0,
                            Omat,
                            ra_sun_in_deg,
                            dec_sun_in_deg,
                            ra[i],
                            dec[i],
                        )
                    )
    return exp_obs_dip


def exp_obs_quadrupole(
    v_bulk,
    ra_bulk,
    dec_bulk,
    a11,
    a12,
    a13,
    a22,
    a23,
    M,
    H0,
    Omat,
    v_sun,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec,
    mu,
    z,
    horstmann,
    is_cal=None,
    ceph=None,
    no_cepheid=None,
):
    number_of_elements = len(mu)
    exp_obs_quad = np.zeros(number_of_elements)
    alpha_matrix = symmetric_traceless_matrix(a11, a12, a13, a22, a23)
    if horstmann:
        for i in range(0, number_of_elements):
            exp_obs_quad[i] = mu[i] - quadrupole(
                v_bulk,
                ra_bulk,
                dec_bulk,
                H0,
                Omat,
                alpha_matrix,
                v_sun,
                z[i],
                ra_sun_in_deg,
                dec_sun_in_deg,
                ra[i],
                dec[i],
            )
    else:
        if no_cepheid:
            for i in range(0, number_of_elements):
                exp_obs_quad[i] = mu[i] - quadrupole(
                    v_bulk,
                    ra_bulk,
                    dec_bulk,
                    H0,
                    Omat,
                    alpha_matrix,
                    v_sun,
                    z[i],
                    ra_sun_in_deg,
                    dec_sun_in_deg,
                    ra[i],
                    dec[i],
                )
        else:
            for i in range(0, number_of_elements):
                if is_cal[i] == 1:
                    exp_obs_quad[i] = mu[i] + M - ceph[i]
                    # exp_obs_dip[i]=mu[i]-dipole(velocity, z7[i],73.4,0.338,scalar[i])
                else:
                    exp_obs_quad[i] = (
                        mu[i]
                        + M
                        - quadrupole(
                            v_bulk,
                            ra_bulk,
                            dec_bulk,
                            H0,
                            Omat,
                            alpha_matrix,
                            v_sun,
                            z[i],
                            ra_sun_in_deg,
                            dec_sun_in_deg,
                            ra[i],
                            dec[i],
                        )
                    )
    return exp_obs_quad


def exp_obs_bulk(
    v_bulk,
    ra_bulk,
    dec_bulk,
    velocity,
    M,
    H0,
    Omat,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec,
    mu,
    z,
    horstmann,
    is_cal=None,
    ceph=None,
    no_cepheid=None,
):
    number_of_elements = len(mu)
    exp_obs_quad = np.zeros(number_of_elements)
    if horstmann:
        for i in range(0, number_of_elements):
            exp_obs_quad[i] = mu[i] - bulk(
                v_bulk,
                ra_bulk,
                dec_bulk,
                H0,
                Omat,
                velocity,
                z[i],
                ra_sun_in_deg,
                dec_sun_in_deg,
                ra[i],
                dec[i],
            )
    else:
        if no_cepheid:
            for i in range(0, number_of_elements):
                exp_obs_quad[i] = mu[i] - bulk(
                    v_bulk,
                    ra_bulk,
                    dec_bulk,
                    H0,
                    Omat,
                    velocity,
                    z[i],
                    ra_sun_in_deg,
                    dec_sun_in_deg,
                    ra[i],
                    dec[i],
                )
        else:
            for i in range(0, number_of_elements):
                if is_cal[i] == 1:
                    exp_obs_quad[i] = mu[i] + M - ceph[i]
                    # exp_obs_dip[i]=mu[i]-dipole(velocity, z7[i],73.4,0.338,scalar[i])
                else:
                    exp_obs_quad[i] = (
                        mu[i]
                        + M
                        - bulk(
                            v_bulk,
                            ra_bulk,
                            dec_bulk,
                            H0,
                            Omat,
                            velocity,
                            z[i],
                            ra_sun_in_deg,
                            dec_sun_in_deg,
                            ra[i],
                            dec[i],
                        )
                    )
    return exp_obs_quad


"""
I define in the following the function for the redshift corrections"""


def z_sun(v, ra_sun_in_deg, dec_sun_in_deg, ra, dec):
    # for dimensionale purpose I have to put 1/c
    c = 299792.458
    vsun = v * factor_scalar(ra_sun_in_deg, dec_sun_in_deg, ra, dec)
    return ((1 + (-vsun / c)) / (1 - (-vsun / c))) ** (1 / 2) - 1


def z_pec(
    z,
    v_bulk,
    ra_bulk,
    dec_bulk,
    H0,
    Omat,
    alpha_matrix,
    ra,
    dec,
):
    # for dimensional purpose I have to put 1/c
    c = 299792.458
    scalar_bulk = factor_scalar(ra_bulk, dec_bulk, ra, dec)
    scalar_bulk_matrix = factor_scalar_matrix(alpha_matrix, ra, dec, ra, dec)
    v_pec = prefactor_z(z, H0, Omat) * (v_bulk * scalar_bulk + scalar_bulk_matrix)
    return (((1 + (v_pec / c))) / (1 - (v_pec / c))) ** (1 / 2) - 1

def z_pec_monopole_bulk_quadrupole(
    z,
    beta,
    v_bulk,
    ra_bulk,
    dec_bulk,
    H0,
    Omat,
    alpha_matrix,
    ra,
    dec,
):
    # for dimensional purpose I have to put 1/c
    c = 299792.458
    scalar_bulk = factor_scalar(ra_bulk, dec_bulk, ra, dec)
    scalar_bulk_matrix = factor_scalar_matrix(alpha_matrix, ra, dec, ra, dec)
    v_pec = prefactor_z(z, H0, Omat) * (beta + v_bulk * scalar_bulk + scalar_bulk_matrix)
    return (((1 + (v_pec / c))) / (1 - (v_pec / c))) ** (1 / 2) - 1

def z_pec_monopole_bulk(
    z,
    beta,
    v_bulk,
    ra_bulk,
    dec_bulk,
    H0,
    Omat,
    ra,
    dec,
):
    # for dimensional purpose I have to put 1/c
    c = 299792.458
    scalar_bulk = factor_scalar(ra_bulk, dec_bulk, ra, dec)
    v_pec = prefactor_z(z, H0, Omat) * (beta+ v_bulk * scalar_bulk)
    return (((1 + (v_pec / c))) / (1 - (v_pec / c))) ** (1 / 2) - 1

def z_monopole_bulk(
    z,
    beta,
    v_bulk,
    ra_bulk,
    dec_bulk,
    H0,
    Omat,
    v,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec,
):
    return (
        (1 + z_cmb_cross(z, v, ra_sun_in_deg, dec_sun_in_deg, ra, dec))
        / (
            1
            + z_pec_monopole_bulk(
                z,
                beta,
                v_bulk,
                ra_bulk,
                dec_bulk,
                H0,
                Omat,
                ra,
                dec,
            )
        )
    ) - 1

def z_monopole_bulk_quadrupole(
    z,
    beta,
    v_bulk,
    ra_bulk,
    dec_bulk,
    H0,
    Omat,
    alpha_matrix,
    v,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec,
):
    return (
        (1 + z_cmb_cross(z, v, ra_sun_in_deg, dec_sun_in_deg, ra, dec))
        / (
            1
            + z_pec_monopole_bulk_quadrupole(
                z,
                beta,
                v_bulk,
                ra_bulk,
                dec_bulk,
                H0,
                Omat,
                alpha_matrix,
                ra,
                dec)
        )
    ) - 1


def z_cmb_cross(z, v, ra_sun_in_deg, dec_sun_in_deg, ra, dec):
    return ((1 + z) / (1 + z_sun(v, ra_sun_in_deg, dec_sun_in_deg, ra, dec))) - 1


def z_monopole_dipole_cross(
    z,
    z_monopole,
    v_0,
    ra_0,
    dec_0,
    ra,
    dec,
):
    return (
        (1 + z_cmb_cross(z, v_0, ra_0, dec_0, ra, dec))
        / (1+z_monopole)) - 1

def z_agnostic_monopole_dipole_quadrupole_cross(
    z,
    z_monopole,
    v_0,
    ra_0,
    dec_0,
    alpha_matrix, 
    ra,
    dec,       
):
    return ((1 + z) / ((1 + z_sun(v_0, ra_0, dec_0, ra, dec))*(1+z_monopole+
            z_agnostic_correction_quadrupole(alpha_matrix,ra,dec,)))) - 1

'''
def z_agnostic_monopole_quadrupole_cross(
    z,
    z_monopole,
    alpha_matrix, 
    ra,
    dec,
):
    """we put the quadrupole correction together with z_0  maybe I won't use it"""
    return (
        (1+z)/
        (1 + z_monopole+z_agnostic_correction_quadrupole(alpha_matrix, ra, dec)))-1
'''

def z_agnostic_correction_quadrupole(
    alpha_matrix,
    ra,
    dec,
):
    """it is the equaivalent of def z_p of the Low_ultipoles paper"""
    """we put the quadrupole correction together with z_0"""
    # for dimensional purpose I have to put 1/c
    c = 299792.458
    scalar_dipole_matrix = factor_scalar_matrix(alpha_matrix, ra, dec, ra, dec)
    v_pec = (scalar_dipole_matrix)
    #then I put the minus in order to be consistent with an agnostic approach but above with the minus for the quadrupole in order to obtain the same result as the previous paper 
    return (((1 + (v_pec / c))) / (1 - (v_pec / c))) ** (1 / 2) - 1



def z_quadrupole(
    z,
    v_bulk,
    ra_bulk,
    dec_bulk,
    H0,
    Omat,
    alpha_matrix,
    v,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec,
):
    return (
        (1 + z_cmb_cross(z, v, ra_sun_in_deg, dec_sun_in_deg, ra, dec))
        / (
            1
            + z_pec(
                z,
                v_bulk,
                ra_bulk,
                dec_bulk,
                H0,
                Omat,
                alpha_matrix,
                ra,
                dec,
            )
        )
    ) - 1


def exp_obs_z_quadrupole_corrected_no_M(
    v_bulk,
    ra_bulk,
    dec_bulk,
    a11,
    a12,
    a13,
    a22,
    a23,
    H0,
    Omat,
    v_sun,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec,
    mu,
    z):
    alpha_matrix = symmetric_traceless_matrix(a11, a12, a13, a22, a23)
    number_of_elements = len(mu)
    exp_obs_mon = np.zeros(number_of_elements)

    for i in range(0, number_of_elements):
        exp_obs_mon[i] = mu[i] - monopole(
            z_quadrupole(
                z[i],
                v_bulk,
                ra_bulk,
                dec_bulk,
                H0,
                Omat,
                alpha_matrix,
                v_sun,
                ra_sun_in_deg,
                dec_sun_in_deg,
                ra[i],
                dec[i],
            ),
            H0,
            Omat,
        )

    return exp_obs_mon



def exp_obs_z_quadrupole_corrected(
    v_bulk,
    ra_bulk,
    dec_bulk,
    a11,
    a12,
    a13,
    a22,
    a23,
    M,
    H0,
    Omat,
    v_sun,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec,
    mu,
    z,
    horstmann,
    is_cal=None,
    ceph=None,
    no_cepheid=None,
):
    c = 0
    alpha_matrix = symmetric_traceless_matrix(a11, a12, a13, a22, a23)
    number_of_elements = len(mu)
    exp_obs_mon = np.zeros(number_of_elements)
    if horstmann:
        for i in range(0, number_of_elements):
            exp_obs_mon[i] = mu[i] - monopole(
                z_quadrupole(
                    z[i],
                    v_bulk,
                    ra_bulk,
                    dec_bulk,
                    H0,
                    Omat,
                    alpha_matrix,
                    v_sun,
                    ra_sun_in_deg,
                    dec_sun_in_deg,
                    ra[i],
                    dec[i],
                ),
                H0,
                Omat,
            )
    else:
        if no_cepheid:
            for i in range(0, number_of_elements):
                exp_obs_mon[i] = mu[i] - monopole(
                    z_quadrupole(
                        z[i],
                        v_bulk,
                        ra_bulk,
                        dec_bulk,
                        H0,
                        Omat,
                        alpha_matrix,
                        v_sun,
                        ra_sun_in_deg,
                        dec_sun_in_deg,
                        ra[i],
                        dec[i],
                    ),
                    H0,
                    Omat,
                )
        else:
            for i in range(0, number_of_elements):
                # print(z_quadrupole_corrected(v_bulk, ra_bulk, dec_bulk, H0, Omat, alpha_matrix, v_sun, z[i], ra_sun_in_deg, dec_sun_in_deg, ra, dec))

                if is_cal[i] == 1:
                    c = c + 1
                    exp_obs_mon[i] = mu[i] + M - ceph[i]
                else:
                    exp_obs_mon[i] = (
                        mu[i]
                        + M
                        - monopole(
                            z_quadrupole(
                                z[i],
                                v_bulk,
                                ra_bulk,
                                dec_bulk,
                                H0,
                                Omat,
                                alpha_matrix,
                                v_sun,
                                ra_sun_in_deg,
                                dec_sun_in_deg,
                                ra[i],
                                dec[i],
                            ),
                            H0,
                            Omat,
                        )
                    )
    return exp_obs_mon


def exp_obs_z_monopole_bulk_corrected(
    beta,
    v_bulk,
    ra_bulk,
    dec_bulk,
    M,
    H0,
    Omat,
    v_sun,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec,
    mu,
    z,
    horstmann,
    is_cal=None,
    ceph=None,
    no_cepheid=None,
):
    c = 0
    number_of_elements = len(mu)
    exp_obs_mon = np.zeros(number_of_elements)
    if horstmann:
        for i in range(0, number_of_elements):
            exp_obs_mon[i] = mu[i] - monopole(
                z_monopole_bulk(
                    z[i],
                    beta,
                    v_bulk,
                    ra_bulk,
                    dec_bulk,
                    H0,
                    Omat,
                    v_sun,
                    ra_sun_in_deg,
                    dec_sun_in_deg,
                    ra[i],
                    dec[i],
                ),
                H0,
                Omat,
            )
    else:
        if no_cepheid:
            for i in range(0, number_of_elements):
                exp_obs_mon[i] = mu[i] - monopole(
                z_monopole_bulk(
                    z[i],
                    beta,
                    v_bulk,
                    ra_bulk,
                    dec_bulk,
                    H0,
                    Omat,
                    v_sun,
                    ra_sun_in_deg,
                    dec_sun_in_deg,
                    ra[i],
                    dec[i],
                    ),
                    H0,
                    Omat,
                )
        else:
            for i in range(0, number_of_elements):
                # print(z_quadrupole_corrected(v_bulk, ra_bulk, dec_bulk, H0, Omat, alpha_matrix, v_sun, z[i], ra_sun_in_deg, dec_sun_in_deg, ra, dec))

                if is_cal[i] == 1:
                    c = c + 1
                    exp_obs_mon[i] = mu[i] + M - ceph[i]
                else:
                    exp_obs_mon[i] = (
                        mu[i]
                        + M
                        - monopole(
                z_monopole_bulk(
                    z[i],
                    beta,
                    v_bulk,
                    ra_bulk,
                    dec_bulk,
                    H0,
                    Omat,
                    v_sun,
                    ra_sun_in_deg,
                    dec_sun_in_deg,
                    ra[i],
                    dec[i],
                            ),
                            H0,
                            Omat,
                        )
                    )
    return exp_obs_mon

def exp_obs_z_monopole_bulk_quadrupole_corrected(
    beta,
    v_bulk,
    ra_bulk,
    dec_bulk,
    a11,
    a12,
    a13,
    a22,
    a23,
    M,
    H0,
    Omat,
    v_sun,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec,
    mu,
    z,
    horstmann,
    is_cal=None,
    ceph=None,
    no_cepheid=None,
):
        
    c = 0
    alpha_matrix = symmetric_traceless_matrix(a11, a12, a13, a22, a23)    
    number_of_elements = len(mu)
    exp_obs_mon = np.zeros(number_of_elements)
    if horstmann:
        for i in range(0, number_of_elements):
            exp_obs_mon[i] = mu[i] - monopole(
                z_monopole_bulk_quadrupole(
                    z[i],
                    beta,
                    v_bulk,
                    ra_bulk,
                    dec_bulk,
                    H0,
                    Omat,
                    alpha_matrix,
                    v_sun,
                    ra_sun_in_deg,
                    dec_sun_in_deg,
                    ra[i],
                    dec[i],
                ),
                H0,
                Omat,
            )
    else:
        if no_cepheid:
            for i in range(0, number_of_elements):
                exp_obs_mon[i] = mu[i] - monopole(
                z_monopole_bulk_quadrupole(
                    z[i],
                    beta,
                    v_bulk,
                    ra_bulk,
                    dec_bulk,
                    H0,
                    Omat,
                    alpha_matrix,
                    v_sun,
                    ra_sun_in_deg,
                    dec_sun_in_deg,
                    ra[i],
                    dec[i],
                    ),
                    H0,
                    Omat,
                )
        else:
            for i in range(0, number_of_elements):
                # print(z_quadrupole_corrected(v_bulk, ra_bulk, dec_bulk, H0, Omat, alpha_matrix, v_sun, z[i], ra_sun_in_deg, dec_sun_in_deg, ra, dec))

                if is_cal[i] == 1:
                    c = c + 1
                    exp_obs_mon[i] = mu[i] + M - ceph[i]
                else:
                    exp_obs_mon[i] = (
                        mu[i]
                        + M
                        - monopole(
                z_monopole_bulk_quadrupole(
                    z[i],
                    beta,
                    v_bulk,
                    ra_bulk,
                    dec_bulk,
                    H0,
                    Omat,
                    alpha_matrix,
                    v_sun,
                    ra_sun_in_deg,
                    dec_sun_in_deg,
                    ra[i],
                    dec[i],
                            ),
                            H0,
                            Omat,
                        )
                    )
    return exp_obs_mon

def exp_obs_z_monopole_bulk_quadrupole_corrected_dl_extra_corrected(
    beta,
    v_bulk,
    ra_bulk,
    dec_bulk,
    a11,
    a12,
    a13,
    a22,
    a23,
    M,
    H0,
    Omat,
    v_sun,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec,
    mu,
    z,
    horstmann,
    is_cal=None,
    ceph=None,
    no_cepheid=None,
):
        
    c = 0
    alpha_matrix = symmetric_traceless_matrix(a11, a12, a13, a22, a23)    
    number_of_elements = len(mu)
    exp_obs_mon = np.zeros(number_of_elements)
    if horstmann:
        for i in range(0, number_of_elements):
            exp_obs_mon[i] = mu[i] - monopole_dl_extra_corrected(beta, v_bulk, ra_bulk, dec_bulk, H0, Omat, alpha_matrix,v_sun,
                                    ra_sun_in_deg,dec_sun_in_deg,ra[i],dec[i],z[i])
    else:
        if no_cepheid:
            for i in range(0, number_of_elements):
                exp_obs_mon[i] = mu[i] - monopole_dl_extra_corrected(beta, v_bulk, ra_bulk, dec_bulk, H0, Omat, alpha_matrix,v_sun,
                                    ra_sun_in_deg,dec_sun_in_deg,ra[i],dec[i],z[i])
        else:
            for i in range(0, number_of_elements):
                # print(z_quadrupole_corrected(v_bulk, ra_bulk, dec_bulk, H0, Omat, alpha_matrix, v_sun, z[i], ra_sun_in_deg, dec_sun_in_deg, ra, dec))

                if is_cal[i] == 1:
                    c = c + 1
                    exp_obs_mon[i] = mu[i] + M - ceph[i]
                else:
                    exp_obs_mon[i] = mu[i] + M - monopole_dl_extra_corrected(beta, v_bulk, ra_bulk, dec_bulk, H0, Omat, alpha_matrix,v_sun,
                                    ra_sun_in_deg,dec_sun_in_deg,ra[i],dec[i],z[i])
    return exp_obs_mon



def exp_obs_z_dipole_corrected(
    M,
    H0,
    Omat,
    velocity,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec,
    mu,
    z,
    horstmann,
    is_cal=None,
    ceph=None,
    no_cepheid=None,
):
    c = 0
    number_of_elements = len(mu)
    exp_obs_mon = np.zeros(number_of_elements)
    if horstmann:
        for i in range(0, number_of_elements):
            exp_obs_mon[i] = mu[i] - monopole(z[i], H0, Omat)

    else:
        if no_cepheid:
            for i in range(0, number_of_elements):
                exp_obs_mon[i] = mu[i] - monopole(
                    z_dipole(
                        z[i], velocity, ra_sun_in_deg, dec_sun_in_deg, ra[i], dec[i]
                    ),
                    H0,
                    Omat,
                )
        else:
            for i in range(0, number_of_elements):
                # print(z_quadrupole_corrected(v_bulk, ra_bulk, dec_bulk, H0, Omat, alpha_matrix, v_sun, z[i], ra_sun_in_deg, dec_sun_in_deg, ra, dec))

                if is_cal[i] == 1:
                    c = c + 1
                    exp_obs_mon[i] = mu[i] + M - ceph[i]
                else:
                    exp_obs_mon[i] = (
                        mu[i]
                        + M
                        - monopole(
                            z_dipole(
                                z[i],
                                velocity,
                                ra_sun_in_deg,
                                dec_sun_in_deg,
                                ra[i],
                                dec[i],
                            ),
                            H0,
                            Omat,
                        )
                    )
    return exp_obs_mon


# let's see how to directly extimated in case of linearization


def velocity_extimated(
    H0,
    Omat,
    mu,
    z,
    horstmann,
    ra_sun_in_deg,
    dec_sun_in_deg,
    ra,
    dec,
    inversed_covariance,
    is_cal=None,
    ceph_dis=None,
    no_cepheid=None,
):
    scalar = factor_scalar(ra_sun_in_deg, dec_sun_in_deg, ra, dec)
    # return np.sum(z)
    errors_in_diagonal_approximation_squared = np.diag(inversed_covariance)
    number_of_elements = len(z)
    numerator = np.zeros(number_of_elements)
    denominator = np.zeros(number_of_elements)
    for i in range(0, number_of_elements):
        numerator[i] = (
            scalar[i]
            * (
                exp_obs_monopole(
                    H0, Omat, mu, z, horstmann, is_cal, ceph_dis, no_cepheid
                )[i]
                * (1 + z[i]) ** 2
            )
            / (
                dl_monopole(z[i], H0, Omat)
                * H0
                * m.sqrt(Omat * (1 + z[i]) ** 3 + (1 - Omat))
                * errors_in_diagonal_approximation_squared[i]
            )
        )
        denominator[i] = (scalar[i] ** 2 * 5 * (1 + z[i]) ** 4) / (
            np.log(10)
            * dl_monopole(z[i], H0, Omat) ** 2
            * H0**2
            * (Omat * (1 + z[i]) ** 3 + (1 - Omat))
            * errors_in_diagonal_approximation_squared[i]
        )

    return sum(numerator) / sum(denominator)


# per come ho definito dl_monopole, e' una funzione solo per scalare. Questo e' il modo migliore cosi ci evitiamo il ciclo for che allunga tanto il lavoro


# Definiamo il filtro da applicare alla funzione


def filter(data_file, inverted_covariance_matrix, boolean_filter):
    # I prepare the file for writing the inverse covariance
    number_of_elements = len(data_file)
    filtered_file = list()
    elements_to_skip = list()
    for i in range(0, number_of_elements):
        if boolean_filter[i]:
            elements_to_skip.append(i)
            continue
        else:
            filtered_file.append(data_file[i])

    """You cannot delete multiple dimensions (such as rows and columns) at once with np.delete(). 
    If you want to delete different dimensions, repeat np.delete()"""

    filtered_covariance_matrix = np.delete(
        np.delete(inverted_covariance_matrix, elements_to_skip, 0), elements_to_skip, 1
    )

    # the third argument in np.delete refers to the fact that it a row (index 0) or a column(index 1)

    return np.array(filtered_file), filtered_covariance_matrix


def filter_z(
    data_file,
    inverted_covariance_matrix,
    redshift,
    down_limit_redshift,
    upper_limit_redshift,
):
    # I prepare the file for writing the inverse covariance
    number_of_elements = len(data_file)
    filtered_file = list()
    elements_to_skip = list()
    for i in range(0, number_of_elements):
        if redshift[i] < down_limit_redshift or redshift[i] > upper_limit_redshift:
            elements_to_skip.append(i)
            continue
        else:
            filtered_file.append(data_file[i])

    """You cannot delete multiple dimensions (such as rows and columns) at once with np.delete(). 
    If you want to delete different dimensions, repeat np.delete()"""

    filtered_covariance_matrix = np.delete(
        np.delete(inverted_covariance_matrix, elements_to_skip, 0), elements_to_skip, 1
    )

    # the third argument in np.delete refers to the fact that it a row (index 0) or a column(index 1)

    return np.array(filtered_file), filtered_covariance_matrix


def filter_z_removing_cepheid(
    data_file,
    inverted_covariance_matrix,
    redshift,
    is_cal,
    down_limit_redshift,
    upper_limit_redshift,
):
    """with this functin I filter keeping all the cepheid"""

    # I prepare the file for writing the inverse covariance
    number_of_elements = len(data_file)
    filtered_file = list()
    elements_to_skip = list()
    for i in range(0, number_of_elements):
        if (
            redshift[i] < down_limit_redshift
            or redshift[i] > upper_limit_redshift
            or is_cal[i] == 1
        ):
            elements_to_skip.append(i)
            continue
        else:
            filtered_file.append(data_file[i])

    """You cannot delete multiple dimensions (such as rows and columns) at once with np.delete(). 
    If you want to delete different dimensions, repeat np.delete()"""

    filtered_covariance_matrix = np.delete(
        np.delete(inverted_covariance_matrix, elements_to_skip, 0), elements_to_skip, 1
    )

    # the third argument in np.delete refers to the fact that it a row (index 0) or a column(index 1)

    return np.array(filtered_file), filtered_covariance_matrix


def filter_z_cepheid(
    data_file,
    inverted_covariance_matrix,
    redshift,
    is_cal,
    down_limit_redshift,
    upper_limit_redshift,
):
    # I prepare the file for writing the inverse covariance
    number_of_elements = len(data_file)
    filtered_file = list()
    elements_to_skip = list()
    for i in range(0, number_of_elements):
        if (
            redshift[i] < down_limit_redshift or redshift[i] > upper_limit_redshift
        ) and is_cal[i] == 0:
            elements_to_skip.append(i)
            continue
        else:
            filtered_file.append(data_file[i])

    """You cannot delete multiple dimensions (such as rows and columns) at once with np.delete(). 
    If you want to delete different dimensions, repeat np.delete()"""

    filtered_covariance_matrix = np.delete(
        np.delete(inverted_covariance_matrix, elements_to_skip, 0), elements_to_skip, 1
    )

    # the third argument in np.delete refers to the fact that it a row (index 0) or a column(index 1)

    return np.array(filtered_file), filtered_covariance_matrix
