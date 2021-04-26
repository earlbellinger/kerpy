#### Author: Earl Bellinger ( bellinger@phys.au.dk ) 
#### Stellar Astrophysics Centre, Aarhus University, Denmark 

import sys
import argparse
import os

### Parse command line arguments 
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Computes Eulerian linear adiabatic oscillation mode "    +\
        "structure kernels. Requires a stellar structure file (amdl), "   +\
        "a file containing eigenfunctions (amde) e.g. from ADIPLS, "      +\
        "and a file containing the derivatives of the first adiabatic "   +\
        "exponent (Gamma_1) from the equation of state. These "           +\
        "derivatives can in principle come from an FGONG file, but note " +\
        "that MESA does not compute them.")
parser.add_argument('--amdl',   default='profile1-freqs.model', 
    type=str, help='filename for (possibly redistributed) structure file')
parser.add_argument('--amde',   default='profile1-freqs.amde', 
    type=str, help='filename for eigenfunctions file')
parser.add_argument('--gamder', default='gamder.out', 
    type=str, help='Gamma_1 derivatives filename')
parser.add_argument('--fgong',  default='profile1.FGONG', 
    type=str, help='FGONG containing Gamma_1 derivatives')
parser.add_argument('--eigen',  default=False, action='store_true',
    help='Calculate homogeneous eigenvalue')
args = parser.parse_args(sys.argv[1:])
amdl_fname   = args.amdl 
amde_fname   = args.amde 
gamder_fname = args.gamder 
fgong_fname  = args.fgong 
solve_lambda = args.eigen

if not os.path.exists(amdl_fname):
    print("Error: invalid path to amdl file", amdl_fname)
    sys.exit(1)

if not os.path.exists(amde_fname):
    print("Error: invalid path to amde file", amde_fname)
    sys.exit(1)

if not os.path.exists(gamder_fname) and\
   not os.path.exists(fgong_fname):
    print("Error: invalid path to gamder file", gamder_fname)
    print("Error: invalid path to FGONG file",  fgong_fname)
    print("Need either FGONG file or external file containing EOS "         +\
          "derivatives. Note that MESA FGONG files do not work as MESA does"+\
          " not output the EOS derivatives.") 
    sys.exit(1)

################################################################################
################################################################################
################################################################################

### Import packages for calculations 
import numpy as np
import pandas as pd
from scipy.integrate   import solve_bvp
from scipy.interpolate import InterpolatedUnivariateSpline
from tomso import fgong, adipls
from tomso.utils import integrate, complement 
from tqdm import tqdm

# we divide by 0 here, and we like it 
np.seterr(divide='ignore', invalid='ignore')


## load from amdl file because we redistribute the mesh 
# ref: Notes on adiabatic oscillation programme 
# Section 5: Equilibrium model variables 
# https://users-phys.au.dk/jcd/adipack.v0_3/
amdl = adipls.load_amdl(amdl_fname, return_object=True) 
G    = amdl.G # gravitational constant 
M, R, P_c, rho_c    = amdl.D[:4]  # mass, radius, central density and pressure 
x, qv, Vg, G1, A, U = amdl.A.T    # dimensionless structure 
r   = x*R                         # distance from center 
m   = qv*x**3*M                   # mass coordinate 
g   = G*m/r**2                    # local gravitational acceleration 
rho = qv*U*M/4./np.pi/R**3        # density 
P   = G*m*rho/G1/r/Vg             # pressure 
drho_dr = -(A+Vg)*rho/r           # density gradient 

# fix irregularities at the centre 
rho    [x==0] = rho_c
P      [x==0] = P_c
g      [x==0] = 0
drho_dr[x==0] = 0

cs2 = G1*P/rho # square of the adiabatic sound speed


## solve the homogeneous eigenvalue problem for lambda ~ 1
# ref: Bellinger et al. submitted, Appendix eq. (A9)
# var_ denotes a spline interpolation of var 
rho_ = InterpolatedUnivariateSpline(r, rho)
P_   = InterpolatedUnivariateSpline(r, P)

rho_hat = InterpolatedUnivariateSpline(x, rho * (M/R**3)**-1)
P_hat   = InterpolatedUnivariateSpline(x, P   * (G*M**2/R**4)**-1)

if solve_lambda:
    def fun(T, y, p):
        t = T.copy()
        t[t==0] = 1e-15
        #y1 = t**2 * rho_(t) * y[1]
        #y2 = -p * 4*np.pi*G*rho_(t) / (t**2 * P_(t)) * y[0]
        y1 = t**2 * rho_hat(t) * y[1]
        y2 = -p * 4*np.pi*rho_hat(t) / (t**2 * P_hat(t)) * y[0]
        return np.vstack((y1, y2))
    
    result = solve_bvp(fun=fun, 
        bc=lambda ya, yb, p: np.array([ya[0], yb[0], yb[1]-1]), 
        x=x,#r,
        p=(1,),
        tol=1e-6,
        max_nodes=1e7,
        y=np.ones((2, len(r))))
    
    with open('lambda.dat', 'w') as f:
        f.write('%f' % result.p)
    
    psi   = InterpolatedUnivariateSpline(result.x, result.y[0])(x)
    K_psi = P_hat(x) * np.gradient(psi / P_hat(x), x)
    #K_psi = P * np.gradient(psi / P, r)
    pd.DataFrame({'x':x, 'psi':psi, 'K_psi':K_psi}).to_csv('psi_lambda.dat', 
        sep='\t', index=False)

#exit() 

## amde file holds the eigenfunctions 
# ref: ibid., Section 8.4 
# important note: TOMSO uses nfmode=1 ! 
amde = adipls.load_amde(amde_fname, nfmode=1, return_object=True)


## read equation of state derivatives from file, either FGONG or external
if os.path.exists(gamder_fname):
    gamder = pd.read_table(gamder_fname, sep='\\s+',
        names=['x', 'dgam_P', 'dgam_rho', 'dgam_Y'])[::-1]
else:
    gong = fgong.load_fgong(fgong_fname, return_object=True)
    if gong.var.shape <= 25:
        print("Error: FGONG does not contain EOS derivatives")
    
    gamder = pd.DataFrame({
        'x'        : gong.var[:,0] / R, 
        'dgam_rho' : gong.var[:,25],  # (dGamma_1/drho)_P,Y
        'dgam_P'   : gong.var[:,26],  # (dGamma_1/dP)_rho,Y
        'dgam_Y'   : gong.var[:,27]}) # (dGamma_1/dY)_P,rho

# interpolate onto remeshed grid 
dG1rho = InterpolatedUnivariateSpline(gamder['x'], gamder['dgam_rho'])(x)
dG1P   = InterpolatedUnivariateSpline(gamder['x'], gamder['dgam_P'])(x)
dG1Y   = InterpolatedUnivariateSpline(gamder['x'], gamder['dgam_Y'])(x)


## now compute kernels and store them in the following dicts 
K_c2rhos = {}
K_rhoc2s = {}
K_G1rhos = {}
K_rhoG1s = {}
K_uYs    = {}
K_Yus    = {}
K_uG1s   = {}
K_G1us   = {}

# iterate over each eigenmode 
for ii in tqdm(range(len(amde.eigs))): 
    n   = amde.n[ii]
    ell = amde.l[ii]
    L2  = ell*(ell+1)
    
    eig  = amde.eigs[ii]
    y    = eig.T
    x    = y[0]
    xi_r = y[1] * R
    xi_h = y[2] * R / L2
    
    sigma2 = amde.css[ii]['sigma2']         # dimensionless frequency 
    omega = np.sqrt(sigma2*G*M/R**3)        # angular frequency 
    
    if ell == 0:
        xi_h     = 0.*xi_r  # radial modes have zero horizontal component
        chi      = Vg/x*(y[1]-sigma2/qv/x*y[2])
        dxi_r_dr = chi - 2.*y[1]/x
        dPhi_dr  = -4.*np.pi*G*rho*xi_r
        Phi      = -complement(dPhi_dr, r)  # unused(?)
    else:
        eta      = L2*qv/sigma2
        chi      = Vg/x*(y[1]-y[2]/eta-y[3])
        dxi_r_dr = chi - 2.*y[1]/x + y[2]/x
        dPhi_dr  = -g/x*(y[3] + y[4]) - y[3]*R*(4.*np.pi*G*rho - 2.*g/r)
        Phi      = -g*R*y[3]
    
    Phi_r = Phi/r # Phi/r, Phi/r on the mountain (ref: Grateful Dead)
    
    chi     [x==0] = 0.
    dxi_r_dr[x==0] = 0.
    dPhi_dr [x==0] = 0.
    Phi_r   [x==0] = 0.
    
    S = np.trapz((xi_r**2 + L2*xi_h**2)*rho*r**2, r)
    
    
    ## Kernel pair ($c_s^2, \rho$)
    # ref: InversionKit equation (103)
    # https://lesia.obspm.fr/perso/daniel-reese/spaceinn/inversionkit/
    # download source and look at doc.pdf 
    K_c2rho = rho*cs2*chi**2*r**2 / S / omega**2 / 2.  
    K_rhoc2 = cs2*chi**2 \
        - omega**2*(xi_r**2+L2*xi_h**2) \
        - 2.*g*xi_r*chi \
        - 4.*np.pi*G*complement((2.*rho*chi+xi_r*drho_dr)*xi_r, r) \
        + 2.*g*xi_r*dxi_r_dr \
        + 4.*np.pi*G*rho*xi_r**2 \
        + 2.*(xi_r*dPhi_dr + L2*xi_h*Phi_r)
    K_rhoc2 *= rho*r**2/2./S/omega**2
    K_rhoc2_ = np.copy(K_rhoc2)
    
    # remove complementary function after projection into an orthogonal vector 
    T_rhoc2  = rho*r**2
    alpha    = np.trapz(K_rhoc2*T_rhoc2, r)/np.trapz(T_rhoc2*T_rhoc2, r)
    K_rhoc2 -= alpha*T_rhoc2
    
    
    ## Kernel pair ($Gamma_1, \rho$)
    # ref: InversionKit equation (105)
    K_G1rho = K_c2rho # ez pz 
    int_K_c2rho_rho = integrate(G1 * chi**2 * r**2 / 2. / S / omega**2, r)
    integrand = rho / r**2 * int_K_c2rho_rho
    integrand[x==0] = 0.
    K_rhoG1 = K_rhoc2_ - K_c2rho + \
        g * rho * int_K_c2rho_rho + \
        4 * np.pi * G * rho * r**2 * complement(integrand, r) 
    K_rhoG1_ = np.copy(K_rhoG1)
    
    T_rhoG1  = rho*r**2
    alpha    = np.trapz(K_rhoG1*T_rhoG1, r)/np.trapz(T_rhoG1*T_rhoG1, r)
    K_rhoG1 -= alpha*T_rhoc2
    
    
    # Kernel pair ($u,Y$)
    # ref: Appendix of Thompson & Christensen-Dalsgaard (2002) 
    # https://ui.adsabs.harvard.edu/abs/2002ESASP.485...95T
    K_Yu = dG1Y * K_G1rho # also ez pz
    F = K_rhoG1 + (dG1P + dG1rho) * K_G1rho 
    #   ^ should we use K_rhoG1 with or without its comp. func. removed? 
    
    # var_ denotes a spline interpolation of var 
    F_   = InterpolatedUnivariateSpline(r, F)
    
    def bvp(T, y, eps=1e-15):
        # T is the re-remeshed grid 
        # now recast the second-order system as a first-order system
        t = T.copy()
        t[t==0] = eps
        dy1 = t**2 * rho_(t) * y[1] + F_(t)
        dy2 = -4*np.pi*G*rho_(t) / (t**2 * P_(t)) * y[0]
        return np.vstack((dy1, dy2))
    
    result = solve_bvp(fun=bvp, x=r, y=np.zeros((2, len(r))),
        bc=lambda ya, yb: np.array([ya[0], yb[0]]), # psi(x=0)=psi(x=1)=0
        tol=1e-12, max_nodes=100000)
    
    if not result.success:
        print("Error: Failed to solve for psi for n =", n, "l =", ell)
        K_uY = 0.*r
    else:
        psi = InterpolatedUnivariateSpline(result.x, result.y[0,:])(r)
        K_uY = dG1P * K_G1rho - P * np.gradient(psi / P, r)
    
    
    # Kernel pair ($u,G1$)
    # ref: Appendix of Thompson & Christensen-Dalsgaard (2002) 
    # https://ui.adsabs.harvard.edu/abs/2002ESASP.485...95T
    K_G1u = K_c2rho # also ez pz
    F_ = InterpolatedUnivariateSpline(r, K_rhoc2)
    result = solve_bvp(fun=bvp, x=r, y=np.zeros((2, len(r))),
        bc=lambda ya, yb: np.array([ya[0], yb[0]]), # psi(x=0)=psi(x=1)=0
        tol=1e-12, max_nodes=100000)
    
    if not result.success:
        print("Error: Failed to solve for psi for n =", n, "l =", ell)
        K_uG1 = 0.*r
    else:
        psi = InterpolatedUnivariateSpline(result.x, result.y[0,:])(r)
        K_uG1 = K_c2rho - P * np.gradient(psi / P, r)
    
    if ii == 0:
        nl = 'x'
        K_c2rhos[nl] = x
        K_rhoc2s[nl] = x
        K_G1rhos[nl] = x
        K_rhoG1s[nl] = x
        K_uYs   [nl] = x
        K_Yus   [nl] = x
        K_uG1s  [nl] = x
        K_G1us  [nl] = x
        
        # write complementary function as well 
        nl = 'l.0_n.-1'
        K_c2rhos[nl] = 0.*x
        K_rhoc2s[nl] = R*T_rhoc2
        K_G1rhos[nl] = 0.*x
        K_rhoG1s[nl] = R*T_rhoG1
        K_uYs   [nl] = 0.*x
        K_Yus   [nl] = 0.*x
        K_uG1s  [nl] = 0.*x
        K_G1us  [nl] = 0.*x
    
    nl  = 'l.'+str(int(ell))+'_n.'+str(int(n))
    K_c2rhos[nl] = R*K_c2rho
    K_rhoc2s[nl] = R*K_rhoc2
    K_G1rhos[nl] = R*K_G1rho
    K_rhoG1s[nl] = R*K_rhoG1
    K_uYs   [nl] = R*K_uY
    K_Yus   [nl] = R*K_Yu
    K_uG1s  [nl] = R*K_uG1
    K_G1us  [nl] = R*K_G1u

# write to disk 
pd.DataFrame(K_c2rhos).to_csv('K_c2-rho.dat', sep='\t', index=False)
pd.DataFrame(K_rhoc2s).to_csv('K_rho-c2.dat', sep='\t', index=False)
pd.DataFrame(K_G1rhos).to_csv('K_G1-rho.dat', sep='\t', index=False)
pd.DataFrame(K_rhoG1s).to_csv('K_rho-G1.dat', sep='\t', index=False)
pd.DataFrame(K_uYs)   .to_csv('K_u-Y.dat',    sep='\t', index=False)
pd.DataFrame(K_Yus)   .to_csv('K_Y-u.dat',    sep='\t', index=False)
pd.DataFrame(K_uG1s)  .to_csv('K_u-G1.dat',   sep='\t', index=False)
pd.DataFrame(K_G1us)  .to_csv('K_G1-u.dat',   sep='\t', index=False)
print("Success: wrote kernels to K_c2-rho.dat etc.") 
