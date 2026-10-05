"""Axisymmetric dipole collection through a flat interface and annular pupil.

Propagating water-side far field, ideal aplanatic objective, full detector
collection. No supercritical emission, near-field dipole modification,
aberrations, background or interference. Angles are relative to the optical
axis; NA is n*sin(alpha). Coefficients describe intensity ratios, not calibrated
absolute power.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def annular_interface_abc(na_in=0.38, na_out=1.3, n_sample=1.33,
                          n_immersion=1.51, quadrature_order=128):
    """Return A,B,C for I_psi = K[A+B sin²theta+C sin²theta cos2(phi-psi)].

    Fresnel amplitudes act on local s/p components. The field mapping to a
    uniform Cartesian BFP grid is sqrt(cos(alpha_oil))/cos(alpha_water).
    Integrate its square with rho*d(rho)*d(azimuth). Angular integrals are
    analytic; radial integrals use Gauss-Legendre quadrature. A+B is normalized
    to one, absorbing the common positive factor into K. This normalization
    must not be used to compare absolute throughput between pupils.
    """
    params = np.array([na_in, na_out, n_sample, n_immersion], dtype=float)
    if not np.all(np.isfinite(params)) or not (
        0 <= na_in < na_out < min(n_sample, n_immersion)
    ):
        raise ValueError('Require 0 <= na_in < na_out < both refractive indices.')
    if not isinstance(quadrature_order, (int, np.integer)) or quadrature_order < 16:
        raise ValueError('quadrature_order must be an integer >= 16.')
    nodes, weights = leggauss(quadrature_order)
    rho = na_in + (nodes + 1) * (na_out - na_in) / 2
    sw = rho / n_sample
    cw = np.sqrt(1 - sw**2)
    co = np.sqrt(1 - (rho / n_immersion)**2)
    tp = 2*n_sample*cw / (n_sample*co + n_immersion*cw)
    ts = 2*n_sample*cw / (n_sample*cw + n_immersion*co)
    measure = weights * (na_out-na_in)/2 * 2*np.pi*rho * co/cw**2
    p, s = tp*cw, ts
    # Azimuth averages of the squared x-analyser response to x,y,z dipoles.
    qxx = np.dot(measure, (3*p*p + 2*p*s + 3*s*s)/8)
    qyy = np.dot(measure, (p-s)**2/8)
    qzz = np.dot(measure, (tp*sw)**2/2)
    abc = np.array([qzz, (qxx+qyy)/2-qzz, (qxx-qyy)/2])
    return tuple(abc / (abc[0]+abc[1]))
