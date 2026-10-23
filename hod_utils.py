# hod_utils.py
import numpy as np
import scipy.special as sysp
import gc

class HODFits_BGS(object):
    """ Fitting functions specifying best fit 5-parameter HOD fits 
        from Smith, Grove, Cole et al. (2024) Public Data Release.
    """
    def __init__(self):
        self._A_min, self._B_min, self._C_min, self._D_min = -0.1163295942035853792, -0.5759066407274642252, 0.1437085418067726994, -0.01908335449679073731
        self._A_sig, self._B_sig, self._C_sig, self._D_sig = 0.01832666435246161490, 0.7584865035087475782, 1.103103765754579246, 0.3277787214124110449
        self._A_0,   self._B_0                             = -0.5468895455660652827, -1.673562380257160864
        self._A_1,   self._B_1,   self._C_1,   self._D_1   = 1.121497360473139082, -0.4725625165007949491, 0.09200038944813740405, -0.01141762462160326314
        self._A_al,  self._B_al,  self._C_al               = 1.111866664500087198, 4.473949866395862784, -4.353309870091147893

    def _M_function(self, Mr, A, B, C, D):
        x = Mr + 20
        return (A + 12) + B * x + C * x**2 + D * x**3

    def _M0_function(self, Mr, A, B):
        x = Mr + 20
        logM0 = (A + 11) + B * x
        return np.maximum(logM0, 1.0)

    def _sigma_function(self, Mr, A, B, C, D):
        x = Mr + 20
        return A + (B - A) / (1.0 + np.exp(C * (x + D)))

    def _alpha_function(self, Mr, A, B, C):
        x = Mr + 20
        return A + B**(-x + C)

    def hod_fit(self, param, Mr):
        if param == 'kappa':
            lgMmin = self._M_function(Mr, self._A_min, self._B_min, self._C_min, self._D_min)
            Mcut = 10**lgMmin
            logM0 = self._M0_function(Mr, self._A_0, self._B_0)
            M0 = 10**logM0
            return M0 / Mcut
        elif param == 'M1':
            return 10**self._M_function(Mr, self._A_1, self._B_1, self._C_1, self._D_1)
        elif param == 'siglgM':
            return self._sigma_function(Mr, self._A_sig, self._B_sig, self._C_sig, self._D_sig)
        elif param == 'alpha':
            return self._alpha_function(Mr, self._A_al, self._B_al, self._C_al)
        elif param == 'Mcut':
            return 10**self._M_function(Mr, self._A_min, self._B_min, self._C_min, self._D_min)
        elif param == 'M0':
            return 10**self._M0_function(Mr, self._A_0, self._B_0)
        elif param == 'prs':
            x = Mr + 20.5
            return 0.773 - 0.065 * x + 0.008 * x**2

    def get_hod_params_dict(self, Mr=-18.0):
        """Returns dictionary of fiducial HOD parameters."""
        return dict(
            Mcut=self.hod_fit('Mcut', Mr),
            M1=self.hod_fit('M1', Mr),
            sigma=self.hod_fit('siglgM', Mr),
            alpha=self.hod_fit('alpha', Mr),
            kappa=self.hod_fit('kappa', Mr)
        )


class HODFunctions(HODFits_BGS):
    """ HOD probability distributions and galaxy property functions. """
    def __init__(self):
        super().__init__()
        
    def F_spline(self, x):
        t = x / np.sqrt(6)
        at = np.abs(t)
        piece1 = (8/3)*t - (16/3)*t**3 + 4*t**4
        piece2 = np.sign(t) * (1 - (4/3)*(1 - at)**4)
        piece3 = np.sign(t) * 1.0
        return np.where(at <= 0.5, piece1, np.where(at <= 1, piece2, piece3))

    def F_arg(self, Mr, lgm):
        Mr_sc, lgm_sc = np.isscalar(Mr), np.isscalar(lgm)
        lgm_arr = np.outer(lgm, np.ones(Mr.size)) if ((not Mr_sc) & (not lgm_sc)) else 1.0 * lgm
        return (lgm_arr - np.log10(self.hod_fit('Mcut', Mr))) / self.hod_fit('siglgM', Mr)

    def erf_arg(self, Mr, lgm):
        return self.F_arg(Mr, lgm)

    def fcen_thresh(self, Mr, lgm, hod):
        if hod == 'erf':
            return 0.5 * (1 + sysp.erf(self.erf_arg(Mr, lgm)))
        elif hod == 'BGS':
            return 0.5 * (1 + self.F_spline(self.F_arg(Mr, lgm)))

    def Nsat_thresh(self, Mr, lgm):
        M0 = self.hod_fit('M0', Mr)
        M1 = self.hod_fit('M1', Mr)
        alpha = self.hod_fit('alpha', Mr)
        mhalo = 10**lgm
        Mr_sc, lgm_sc = np.isscalar(Mr), np.isscalar(lgm)

        if Mr_sc & lgm_sc:
            return ((mhalo - M0) / M1)**alpha if mhalo > M0 else 0.0
        else:
            mhalo_arr = np.outer(mhalo, np.ones(Mr.size)) if ((not Mr_sc) & (not lgm_sc)) else 1.0 * mhalo
            Dm = (mhalo_arr - M0) / M1
            Dm[Dm < 0.0] = 0.0
            return Dm**alpha

    def mean_color_sat(self, Mr): return 0.83 - 0.08 * (Mr + 20)
    def mean_color_red(self, Mr): return 0.932 - 0.032 * (Mr + 20)
    def std_color_red(self, Mr): return 0.07 + 0.01 * (Mr + 20)
    def mean_color_blue(self, Mr): return 0.62 - 0.11 * (Mr + 20)
    def std_color_blue(self, Mr): return 0.12 + 0.02 * (Mr + 20)
    def blue_frac(self, Mr): return 0.46 + 0.07 * (Mr + 20)
    def p_red_sat(self, Mr):
        return (self.mean_color_sat(Mr) - self.mean_color_blue(Mr)) / (self.mean_color_red(Mr) - self.mean_color_blue(Mr))


class Mocker(HODFunctions):
    """ Assigns luminosities, colors, and secondary properties to populated mock catalogs. """
    def __init__(self, hod='BGS', seed=42):
        super().__init__()
        self.Mrmax = -18.0
        self.hod = hod
        self.rng = np.random.RandomState(seed=seed)

        dMr = 0.001
        nMr = int((23.5 + self.Mrmax) / dMr)
        self.Mrvals = np.linspace(-23.5, self.Mrmax, nMr)
        self.dMr = self.Mrvals[1] - self.Mrvals[0]

    def assign_central_luminosity(self, lgm_halos):
        n = lgm_halos.size
        Mr_cen = np.zeros(n, dtype=float)
        u = self.rng.rand(n)

        for h in range(n):
            PcenL = self.fcen_thresh(self.Mrvals, lgm_halos[h], self.hod) / self.fcen_thresh(self.Mrmax, lgm_halos[h], self.hod)
            Mr_upp = self.Mrvals[PcenL > u[h]][0]
            Mr_low = self.Mrvals[PcenL <= u[h]][-1]
            Mr_cen[h] = np.max([Mr_upp, Mr_low])
        
        del u
        gc.collect()
        return Mr_cen

    def assign_satellite_luminosities(self, centrals, h):
        Nsat_h = centrals['Nsat'][h]
        lgmhalo = centrals['lgm'][h]
        PsatL = self.Nsat_thresh(self.Mrvals, lgmhalo) / self.Nsat_thresh(self.Mrmax, lgmhalo)
        u = self.rng.rand(Nsat_h)
        return np.array([self.Mrvals[(PsatL >= u[s])][0] for s in range(Nsat_h)])

    def assign_satellite_colors(self, satellites):
        prs = np.clip(self.hod_fit('prs', satellites['Mr']), 0.01, 0.99)
        u = self.rng.rand(satellites.size)
        return np.where(u < prs, 'red', 'blue')