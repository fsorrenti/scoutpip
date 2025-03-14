import getdist

# import chainconsumer
import emcee
import matplotlib.pyplot as plt
import numpy as np
from getdist import plots, MCSamples
import cmath
from astropy import coordinates
#import dipole_utilities as ut
from scipy.linalg import eigh

from scoutpy import dipole_utilities as ut


def flatchain(name_file):
    """how to get the flatchain"""
    reader = emcee.backends.HDFBackend(name_file)
    tau = reader.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    return reader.get_chain(flat=True, discard=burnin)

def flatchain_quadrupole(name_chain):
        #load the chain considering all the supernovae
    flatchain_monopole_bulk_quadrupole=flatchain(name_chain)

    flatchain_monopole_bulk_quadrupole[:,3]=np.rad2deg(np.arccos(flatchain_monopole_bulk_quadrupole[:,3]))-90 


    # Computing the symmetric traceless matrix using custom function ut.symmetric_traceless_matrix
    matrix1 = ut.symmetric_traceless_matrix(
        flatchain_monopole_bulk_quadrupole[:, 4],
        flatchain_monopole_bulk_quadrupole[:, 5],
        flatchain_monopole_bulk_quadrupole[:, 6],
        flatchain_monopole_bulk_quadrupole[:, 7],
        flatchain_monopole_bulk_quadrupole[:, 8],
    )


    # Reshaping the array to (277568, 3, 3) for efficient vectorized operations
    matrix1_reshaped = matrix1.transpose(2, 0, 1)

    # Vectorized eigenvalue calculation for each 3x3 matrix in matrix1_reshaped
    eigenvalues_and_vectors_list = [
        find_eigenvalues_eigenvectors(mat) for mat in matrix1_reshaped
    ]

    # Unpack eigenvalues 
    eigenvalues_list = [eig_and_vec[0] for eig_and_vec in eigenvalues_and_vectors_list]


    # Combine selected columns from flatchain_quadrupole and eigenvalues_list
    flatchain_monopole_bulk_quadrupole_with_eigenvalues = np.hstack(
        (flatchain_monopole_bulk_quadrupole[:, [0, 1, 2,3, 9, 10, 11]], eigenvalues_list)
    )

    return flatchain_monopole_bulk_quadrupole_with_eigenvalues

    


"""plotting feauture"""


def plot_getdist(
    flatchain,
    names_parameter=None,
    truth_values=None,
    label_legend=None,
    name_for_saving=None,
    params=None,
    fisher=None,
    smoothing_1D=None,
    smoothing_2D=None,
    legend_fontsize=None,
    ranges=None,
):
    """
    Given a MCMC chain and a set of parameter names, plot the posterior distribution of the parameters.
    flatchain: MCMC chain of parameter values
    names_parameter: names of parameters (in the same order as in the flatchain array)
    truth_values: true parameter values (if known)
    label_legend: label to include in the legend
    name_for_saving: if specified, save the plot as a file with this name
    fisher: if True, interpret the chain as a Fisher matrix instead of a Markov Chain
    legend_fontsize: font size for the legend (default is 15)
    ranges: dictionary of parameter ranges (optional)
    """

    # Set default values
    legend_fontsize = legend_fontsize or 15
    smoothing_1D = smoothing_1D or 0.5
    smoothing_2D = smoothing_2D or 2.5

    # Calculate ranges if not provided
    if ranges is None:
        ranges = {
            name: (
                chain[:, i].mean() - 3 * chain[:, i].std(), 
                chain[:, i].mean() + 3 * chain[:, i].std()
            )
            for chain in flatchain
            for i, name in enumerate(names_parameter)
        }

    with plt.rc_context({"mathtext.fontset": "cm", "font.family": "serif"}):
        # Create the plot object and set the plot settings
        g = plots.get_subplot_plotter()

        # Set global plotting settings
        g.settings.legend_fontsize = legend_fontsize
        g.settings.axes_fontsize = 18
        g.settings.axes_labelsize = 24
        g.settings.axis_marker_lw = 1.5
        g.settings.axis_marker_color = "black"
        g.settings.alpha_filled_add = 0.7
        g.settings.alpha_factor_contour_lines = 0.7

        # Create list of samples from flatchain
        samples = [
            MCSamples(
                samples=chain,
                names=names_parameter,
                label=label_legend[i] if label_legend else f'Chain {i+1}',
                ranges=ranges,
                settings={
                    "smooth_scale_1D": smoothing_1D,
                    "smooth_scale_2D": smoothing_2D,
                },
            )
            for i, chain in enumerate(flatchain)
        ]

        # Add fisher samples if given
        if fisher is not None:
            samples += fisher

        # Plot triangle plot with or without truth values
        markers = (
            {names_parameter[i]: truth_values[i] for i in range(len(names_parameter))}
            if truth_values is not None
            else None
        )

        g.triangle_plot(
            samples,
            filled=True,
            markers=markers,
            params=params,
            contour_colors=["tab:blue", "tab:orange", "tab:green"],
            contour_ls=["-", "-", "-"],
            contour_lws=[1.5, 1.5, 1.5],
            legend_loc="upper right",
            line_args=[
                {"ls": "-", "lw": "1.5", "color": "tab:blue"},
                {"ls": "-", "lw": "1.5", "color": "tab:orange"},
                {"ls": "-", "lw": "1.5", "color": "tab:green"},
            ],
            bbox_inches='tight'
        )

        # Save plot if name_for_saving is given
        if name_for_saving is not None:
            g.export(name_for_saving)

        plt.show()


def contours_getdist(flatchain, names_parameter):
    """this is for a standard flatchain (no list)"""

    print(
        getdist.mcsamples.MCSamples(
            samples=flatchain,
            names=names_parameter,
            settings={"smooth_scale_1D": 0.5, "smooth_scale_2D": 2.5},
        )
        .getTable(limit=1)
        .tableTex()
    )


"""let's define the tools for the eigenvalues"""


def find_eigenvalues_eigenvectors(matrix):
    # Function to find eigenvalues of a symmetric matrix
    eigenvalues, eigenvectors = eigh(matrix)
    return eigenvalues, eigenvectors


def correcting_flatchain_with_eigenvalues_eigenvectors(flatchain_quadrupole):
    # Correcting the third column of flatchain_quadrupole
    flatchain_quadrupole[:, 2] = np.degrees(np.arccos(flatchain_quadrupole[:, 2])) - 90

    # Computing the symmetric traceless matrix using custom function ut.symmetric_traceless_matrix
    matrix1 = ut.symmetric_traceless_matrix(
        flatchain_quadrupole[:, 3],
        flatchain_quadrupole[:, 4],
        flatchain_quadrupole[:, 5],
        flatchain_quadrupole[:, 6],
        flatchain_quadrupole[:, 7],
    )

    # Reshaping the array to (277568, 3, 3) for efficient vectorized operations
    matrix1_reshaped = matrix1.transpose(2, 0, 1)

    # Vectorized eigenvalue calculation for each 3x3 matrix in matrix1_reshaped
    eigenvalues_and_vectors_list = [
        find_eigenvalues_eigenvectors(mat) for mat in matrix1_reshaped
    ]

    # Unpack eigenvalues and eigenvectors separately
    eigenvalues_list = [eig_and_vec[0] for eig_and_vec in eigenvalues_and_vectors_list]
    eigenvectors_list = [eig_and_vec[1] for eig_and_vec in eigenvalues_and_vectors_list]

    # Combine selected columns from flatchain_quadrupole and eigenvalues_list
    flatchain_quadrupole_with_eigenvalues = np.hstack(
        (flatchain_quadrupole[:, [0, 1, 2, 8, 9, 10]], eigenvalues_list)
    )

    return (
        flatchain_quadrupole_with_eigenvalues,
        np.array(eigenvalues_list),
        np.array(eigenvectors_list),
    )


def converter_eigenvectors(eigenvectors):
    # Extracting eigenvector
    a, b, c = eigenvectors[:, :, 0], eigenvectors[:, :, 1], eigenvectors[:, :, 2]

    # Vectorized Cartesian to Spherical conversion
    cartesian_to_spherical = lambda vec: np.array(
        coordinates.cartesian_to_spherical(vec[:, 0], vec[:, 1], vec[:, 2])
    ).T
    result_array0, result_array1, result_array2 = map(cartesian_to_spherical, [a, b, c])

    # Extracting angles and reducing to South hemisphere only
    extract_and_reduce = lambda result: reduce_to_north_emi_only_angles(
        np.degrees(result[:, 1:])
    )
    final_angles0, final_angles1, final_angles2 = map(
        extract_and_reduce, [result_array0, result_array1, result_array2]
    )

    return final_angles0, final_angles1, final_angles2


def reduce_to_south_emi_only_angles(vector):
    """you can use indexing to select the elements of v1_quadrupole[:,2] that are less than zero and apply the negation operation only to those elements:"""
    # Invert the values in the two columns so that I have ra and then dec
    vector = vector[:, ::-1]

    pos_indices = (
        vector[:, 1] > 0
    )  # neg_indices is a boolean array that contains True values for the elements of v1_quadrupole[:,2] that are less than zero, and False values otherwise.
    vector[pos_indices, 1] = -vector[pos_indices, 1]
    vector[pos_indices, 0] = vector[pos_indices, 0] + 180

    pos_indices_ra = vector[:, 0] > 360  # this is for wrapping between 0 and 360
    vector[pos_indices_ra, 0] = vector[pos_indices_ra, 0] - 360
    return vector


def reduce_to_north_emi_only_angles(vector):
    """you can use indexing to select the elements of v1_quadrupole[:,2] that are less than zero and apply the negation operation only to those elements:"""
    # Invert the values in the two columns so that I have ra and then dec
    vector = vector[:, ::-1]

    neg_indices = (
        vector[:, 1] < 0
    )  # neg_indices is a boolean array that contains True values for the elements of v1_quadrupole[:,2] that are less than zero, and False values otherwise.
    vector[neg_indices, 1] = -vector[neg_indices, 1]
    vector[neg_indices, 0] = vector[neg_indices, 0] + 180

    neg_indices_ra = vector[:, 0] > 360  # this is for wrapping between 0 and 360
    vector[neg_indices_ra, 0] = vector[neg_indices_ra, 0] - 360
    return vector



"""analytical formulations"""

I = complex(0, 1)

csqrt_vectorize = np.vectorize(
    cmath.sqrt
)  # that's a cool way for vectorize the cmath sqrt



def chisq_mcmc(name_file):
    reader = emcee.backends.HDFBackend(name_file)
    tau = reader.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    return -2 * np.max(reader.get_log_prob(flat=True, discard=burnin))

 
"""tools for eigenvectors and eigenvalues analysis"""



def eigenvalue_1(a11, a12, a13, a22, a23):
    "Full analytical formula for the third eigenvalues of a symmetric traceless matrix"

    return (
        -(
            3 * a11**2
            + 3 * a11 * a22
            + 3 * a12**2
            + 3 * a13**2
            + 3 * a22**2
            + 3 * a23**2
        )
        / (
            3
            * (
                27 * a11**2 * a22 / 2
                - 27 * a11 * a12**2 / 2
                + 27 * a11 * a22**2 / 2
                + 27 * a11 * a23**2 / 2
                - 27 * a12**2 * a22 / 2
                - 27 * a12 * a13 * a23
                + 27 * a13**2 * a22 / 2
                + csqrt_vectorize(
                    -4
                    * (
                        3 * a11**2
                        + 3 * a11 * a22
                        + 3 * a12**2
                        + 3 * a13**2
                        + 3 * a22**2
                        + 3 * a23**2
                    )
                    ** 3
                    + (
                        27 * a11**2 * a22
                        - 27 * a11 * a12**2
                        + 27 * a11 * a22**2
                        + 27 * a11 * a23**2
                        - 27 * a12**2 * a22
                        - 54 * a12 * a13 * a23
                        + 27 * a13**2 * a22
                    )
                    ** 2
                )
                / 2
            )
            ** (1 / 3)
        )
        - (
            27 * a11**2 * a22 / 2
            - 27 * a11 * a12**2 / 2
            + 27 * a11 * a22**2 / 2
            + 27 * a11 * a23**2 / 2
            - 27 * a12**2 * a22 / 2
            - 27 * a12 * a13 * a23
            + 27 * a13**2 * a22 / 2
            + csqrt_vectorize(
                -4
                * (
                    3 * a11**2
                    + 3 * a11 * a22
                    + 3 * a12**2
                    + 3 * a13**2
                    + 3 * a22**2
                    + 3 * a23**2
                )
                ** 3
                + (
                    27 * a11**2 * a22
                    - 27 * a11 * a12**2
                    + 27 * a11 * a22**2
                    + 27 * a11 * a23**2
                    - 27 * a12**2 * a22
                    - 54 * a12 * a13 * a23
                    + 27 * a13**2 * a22
                )
                ** 2
            )
            / 2
        )
        ** (1 / 3)
        / 3
    ).real


def eigenvalue_2(a11, a12, a13, a22, a23):
    "Full analytical formula for the third eigenvalues of a symmetric traceless matrix"

    return (
        -(
            3 * a11**2
            + 3 * a11 * a22
            + 3 * a12**2
            + 3 * a13**2
            + 3 * a22**2
            + 3 * a23**2
        )
        / (
            3
            * (-1 / 2 - csqrt_vectorize(3) * I / 2)
            * (
                27 * a11**2 * a22 / 2
                - 27 * a11 * a12**2 / 2
                + 27 * a11 * a22**2 / 2
                + 27 * a11 * a23**2 / 2
                - 27 * a12**2 * a22 / 2
                - 27 * a12 * a13 * a23
                + 27 * a13**2 * a22 / 2
                + csqrt_vectorize(
                    -4
                    * (
                        3 * a11**2
                        + 3 * a11 * a22
                        + 3 * a12**2
                        + 3 * a13**2
                        + 3 * a22**2
                        + 3 * a23**2
                    )
                    ** 3
                    + (
                        27 * a11**2 * a22
                        - 27 * a11 * a12**2
                        + 27 * a11 * a22**2
                        + 27 * a11 * a23**2
                        - 27 * a12**2 * a22
                        - 54 * a12 * a13 * a23
                        + 27 * a13**2 * a22
                    )
                    ** 2
                )
                / 2
            )
            ** (1 / 3)
        )
        - (-1 / 2 - csqrt_vectorize(3) * I / 2)
        * (
            27 * a11**2 * a22 / 2
            - 27 * a11 * a12**2 / 2
            + 27 * a11 * a22**2 / 2
            + 27 * a11 * a23**2 / 2
            - 27 * a12**2 * a22 / 2
            - 27 * a12 * a13 * a23
            + 27 * a13**2 * a22 / 2
            + csqrt_vectorize(
                -4
                * (
                    3 * a11**2
                    + 3 * a11 * a22
                    + 3 * a12**2
                    + 3 * a13**2
                    + 3 * a22**2
                    + 3 * a23**2
                )
                ** 3
                + (
                    27 * a11**2 * a22
                    - 27 * a11 * a12**2
                    + 27 * a11 * a22**2
                    + 27 * a11 * a23**2
                    - 27 * a12**2 * a22
                    - 54 * a12 * a13 * a23
                    + 27 * a13**2 * a22
                )
                ** 2
            )
            / 2
        )
        ** (1 / 3)
        / 3
    ).real


def eigenvalue_3(a11, a12, a13, a22, a23):
    "Full analytical formula for the third eigenvalues of a symmetric traceless matrix"
    return (
        -(
            3 * a11**2
            + 3 * a11 * a22
            + 3 * a12**2
            + 3 * a13**2
            + 3 * a22**2
            + 3 * a23**2
        )
        / (
            3
            * (-1 / 2 + csqrt_vectorize(3) * I / 2)
            * (
                27 * a11**2 * a22 / 2
                - 27 * a11 * a12**2 / 2
                + 27 * a11 * a22**2 / 2
                + 27 * a11 * a23**2 / 2
                - 27 * a12**2 * a22 / 2
                - 27 * a12 * a13 * a23
                + 27 * a13**2 * a22 / 2
                + csqrt_vectorize(
                    -4
                    * (
                        3 * a11**2
                        + 3 * a11 * a22
                        + 3 * a12**2
                        + 3 * a13**2
                        + 3 * a22**2
                        + 3 * a23**2
                    )
                    ** 3
                    + (
                        27 * a11**2 * a22
                        - 27 * a11 * a12**2
                        + 27 * a11 * a22**2
                        + 27 * a11 * a23**2
                        - 27 * a12**2 * a22
                        - 54 * a12 * a13 * a23
                        + 27 * a13**2 * a22
                    )
                    ** 2
                )
                / 2
            )
            ** (1 / 3)
        )
        - (-1 / 2 + csqrt_vectorize(3) * I / 2)
        * (
            27 * a11**2 * a22 / 2
            - 27 * a11 * a12**2 / 2
            + 27 * a11 * a22**2 / 2
            + 27 * a11 * a23**2 / 2
            - 27 * a12**2 * a22 / 2
            - 27 * a12 * a13 * a23
            + 27 * a13**2 * a22 / 2
            + csqrt_vectorize(
                -4
                * (
                    3 * a11**2
                    + 3 * a11 * a22
                    + 3 * a12**2
                    + 3 * a13**2
                    + 3 * a22**2
                    + 3 * a23**2
                )
                ** 3
                + (
                    27 * a11**2 * a22
                    - 27 * a11 * a12**2
                    + 27 * a11 * a22**2
                    + 27 * a11 * a23**2
                    - 27 * a12**2 * a22
                    - 54 * a12 * a13 * a23
                    + 27 * a13**2 * a22
                )
                ** 2
            )
            / 2
        )
        ** (1 / 3)
        / 3
    ).real


def eigenvalues(flatchain1, only_quadrupole=False):
    if only_quadrupole:
        e = np.array(
            [
                eigenvalue_1(
                    flatchain1[:, 0],
                    flatchain1[:, 1],
                    flatchain1[:, 2],
                    flatchain1[:, 3],
                    flatchain1[:, 4],
                ),
                eigenvalue_2(
                    flatchain1[:, 0],
                    flatchain1[:, 1],
                    flatchain1[:, 2],
                    flatchain1[:, 3],
                    flatchain1[:, 4],
                ),
                eigenvalue_3(
                    flatchain1[:, 0],
                    flatchain1[:, 1],
                    flatchain1[:, 2],
                    flatchain1[:, 3],
                    flatchain1[:, 4],
                ),
            ]
        )
    else:
        e = np.array(
            [
                eigenvalue_1(
                    flatchain1[:, 3],
                    flatchain1[:, 4],
                    flatchain1[:, 5],
                    flatchain1[:, 6],
                    flatchain1[:, 7],
                ),
                eigenvalue_2(
                    flatchain1[:, 3],
                    flatchain1[:, 4],
                    flatchain1[:, 5],
                    flatchain1[:, 6],
                    flatchain1[:, 7],
                ),
                eigenvalue_3(
                    flatchain1[:, 3],
                    flatchain1[:, 4],
                    flatchain1[:, 5],
                    flatchain1[:, 6],
                    flatchain1[:, 7],
                ),
            ]
        )

    eigen_quadrupole = e.T.tolist()
    return np.array(eigen_quadrupole)


"""eigenvector stuff"""


def v2(a11, a12, a13, a22, a23, l1):
    """obtaining eigenvector from elements symmetrice traceless matrix"""
    return -(a13 / a12 * v3(a11, a12, a13, a22, a23, l1) + (a11 - l1) / a12)


def v3(a11, a12, a13, a22, a23, l1):
    """obtaining eigenvector from elements symmetrice traceless matrix"""
    return (a12**2 - (a22 - l1) * (a11 - l1)) / (a13 * (a22 - l1) - a23 * a12)


def eigenvector(flatchain, n, only_quadrupole=False):
    eigenvalues_computed = eigenvalues(flatchain, only_quadrupole)

    w2 = v2(
        flatchain[:, 3],
        flatchain[:, 4],
        flatchain[:, 5],
        flatchain[:, 6],
        flatchain[:, 7],
        eigenvalues_computed[:, n],
    )
    w3 = v3(
        flatchain[:, 3],
        flatchain[:, 4],
        flatchain[:, 5],
        flatchain[:, 6],
        flatchain[:, 7],
        eigenvalues_computed[:, n],
    )

    normalization = np.sqrt(1 + w2**2 + w3**2)
    a = coordinates.cartesian_to_spherical(
        1 / normalization, w2 / normalization, w3 / normalization
    )

    return np.array([a[0], a[2].degree, a[1].degree]).T  # 2 is ra and 1 is dec


def eigenvector_cartesian(flatchain, n, only_quadrupole=False):
    eigenvalues_computed = eigenvalues(flatchain, only_quadrupole)

    w2 = v2(
        flatchain[:, 3],
        flatchain[:, 4],
        flatchain[:, 5],
        flatchain[:, 6],
        flatchain[:, 7],
        eigenvalues_computed[:, n],
    )
    w3 = v3(
        flatchain[:, 3],
        flatchain[:, 4],
        flatchain[:, 5],
        flatchain[:, 6],
        flatchain[:, 7],
        eigenvalues_computed[:, n],
    )

    normalization = np.sqrt(1 + w2**2 + w3**2)
    # a = coordinates.cartesian_to_spherical(1/normalization, w2/normalization, w3/normalization)

    return np.array(
        [
            np.ones(len(flatchain)) / normalization,
            w2 / normalization,
            w3 / normalization,
        ]
    ).T  # 2 is ra and 1 is dec


def reduce_to_north_emi(vector):
    """you can use indexing to select the elements of v1_quadrupole[:,2] that are less than zero and apply the negation operation only to those elements:"""

    neg_indices = (
        vector[:, 2] < 0
    )  # neg_indices is a boolean array that contains True values for the elements of v1_quadrupole[:,2] that are less than zero, and False values otherwise.
    vector[neg_indices, 2] = -vector[neg_indices, 2]
    vector[neg_indices, 1] = vector[neg_indices, 1] + 180

    neg_indices_ra = vector[:, 1] > 360  # this is for wrapping between 0 and 360
    vector[neg_indices_ra, 1] = vector[neg_indices_ra, 1] - 360
    return vector


def product_eigenvectors_bulk_vector(
    flatchain, eigenvector1_cartesian, eigenvector2_cartesian, eigenvector3_cartesian
):
    sun_cartesian = np.array(
        spherical_to_cartesian(
            1, np.deg2rad(flatchain[:, 2]), np.deg2rad(flatchain[:, 1])
        )
    ).T
    product = np.empty((len(flatchain), 3))
    for i in range(0, len(flatchain)):
        product[i, 0] = eigenvector1_cartesian[i, :] @ sun_cartesian[i, :]
        product[i, 1] = eigenvector2_cartesian[i, :] @ sun_cartesian[i, :]
        product[i, 2] = eigenvector3_cartesian[i, :] @ sun_cartesian[i, :]

    return product




'''for agnostic analysis purpose'''


def taylor2hubble_c_scaled(flatchain_taylor, monopole=None):
    if monopole is None:
        inverse_a = 1 / flatchain_taylor[:, 1]
        q0 = 1 - flatchain_taylor[:, 2] * inverse_a
        flatchain_taylor[:, 2] = q0
        flatchain_taylor[:, 3] = -1 - flatchain_taylor[:, 3] * inverse_a + q0 * (3 * q0 + 1)
        flatchain_taylor[:, 1] = inverse_a
        flatchain_taylor[:,6]=np.rad2deg(np.arccos(flatchain_taylor[:,6]))-90 
 
    else:
        inverse_a = 1 / flatchain_taylor[:, 0]
        q0 = 1 - flatchain_taylor[:, 1] * inverse_a
        flatchain_taylor[:, 1] = q0
        flatchain_taylor[:, 2] = -1 - flatchain_taylor[:, 2] * inverse_a + q0 * (3 * q0 + 1)
        flatchain_taylor[:, 0] = inverse_a
        flatchain_taylor[:,5]=np.rad2deg(np.arccos(flatchain_taylor[:,5]))-90 

    return flatchain_taylor

def taylor2hubble_c_scaled_simple(flatchain_taylor, monopole=None):

    """without assuming dipole"""
    if monopole is None:
        inverse_a = 1 / flatchain_taylor[:, 1]
        q0 = 1 - flatchain_taylor[:, 2] * inverse_a
        flatchain_taylor[:, 2] = q0
        flatchain_taylor[:, 3] = -1 - flatchain_taylor[:, 3] * inverse_a + q0 * (3 * q0 + 1)
        flatchain_taylor[:, 1] = inverse_a
    else:
        inverse_a = 1 / flatchain_taylor[:, 0]
        q0 = 1 - flatchain_taylor[:, 1] * inverse_a
        flatchain_taylor[:, 1] = q0
        flatchain_taylor[:, 2] = -1 - flatchain_taylor[:, 2] * inverse_a + q0 * (3 * q0 + 1)
        flatchain_taylor[:, 0] = inverse_a
    return flatchain_taylor

def taylor2hubble(flatchain_taylor):
    c = 299792.458
    inverse_a = 1 / flatchain_taylor[:, 0]
    q0 = 1 - flatchain_taylor[:, 1] * inverse_a
    flatchain_taylor[:, 1] = q0
    flatchain_taylor[:, 2] = -1 - flatchain_taylor[:, 2] * inverse_a + q0 * (3 * q0 + 1)
    flatchain_taylor[:, 0] = c * inverse_a
    return flatchain_taylor




def best_fit_mcmc(name_file):
    reader = emcee.backends.HDFBackend(name_file)
    tau = reader.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    flatchain = reader.get_chain(flat=True, discard=burnin)
    log_prob = reader.get_log_prob(flat=True, discard=burnin)

    print((-2 * np.max(log_prob)))  # it must be equal to the chisq
    position_max_log_prob = np.argmax(log_prob)

    print(flatchain[position_max_log_prob])


def plot_mcmc(name_file):
    reader = emcee.backends.HDFBackend(name_file)
    tau = reader.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    flatchain = reader.get_chain(flat=True, discard=burnin)
    chain_pantheon_large = chainconsumer.ChainConsumer().add_chain(
        flatchain,
        walkers=32,
        parameters=[
            "$\Omega_m$",
            "$\Omega_b$",
            "$n_s$",
            "$\sigma_8$",
            "$w_0$",
            "$w_a$",
            "$h$",
        ],
    )
    plt.plot(flatchain[:, :], ",")
    print(chain_pantheon_large.diagnostic.gelman_rubin())
    chain_pantheon_large.plotter.plot(
        truth=[0.32, 0.05, 0.96, 0.81554, -1, 0, 0.67]
    ).savefig(name_file + "plot.png", dpi=300)
    chain_pantheon_large.plotter.plot_distributions(
        truth=[0.32, 0.05, 0.96, 0.81554, -1, 0, 0.67]
    ).savefig(name_file + "_" + "plot_distribution.png", dpi=300)
