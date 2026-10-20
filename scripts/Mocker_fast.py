# --- Canonical luminosity assigner, adopted from this point forward ---
class LuminosityAssigner:
    def __init__(self, Mrmax, seed=42, dMr=0.001):
        self.Mrmax = Mrmax
        self.rng = np.random.RandomState(seed=seed)
        nMr = int((23.5 + self.Mrmax) / dMr)
        self.Mrvals = np.linspace(-23.5, self.Mrmax, nMr)
        self.dMr = self.Mrvals[1] - self.Mrvals[0]

        hod_grid = get_hod_params(self.Mrvals)
        self.logMcut_grid = np.log10(hod_grid['Mcut'])
        self.sigma_grid   = hod_grid['sigma']
        self.M0_grid, self.M1_grid, self.alpha_grid = hod_grid['M0'], hod_grid['M1'], hod_grid['alpha']

        hod_max = get_hod_params(self.Mrmax)
        self.logMcut_max, self.sigma_max = np.log10(hod_max['Mcut']), hod_max['sigma']
        self.M0_max, self.M1_max, self.alpha_max = hod_max['M0'], hod_max['M1'], hod_max['alpha']

    def _fcen_grid(self, lgm):
        erfarg = (lgm - self.logMcut_grid) / self.sigma_grid
        return 0.5 * (1 + sysp.erf(erfarg))

    def _fcen_at_max(self, lgm):
        erfarg = (lgm - self.logMcut_max) / self.sigma_max
        return 0.5 * (1 + sysp.erf(erfarg))

    def assign_central_luminosity(self, lgm_halos):
        Mr_cen = np.zeros(lgm_halos.size, dtype=float)
        u = self.rng.rand(lgm_halos.size)
        Pnorm = self._fcen_at_max(lgm_halos)
        for h in range(lgm_halos.size):
            PcenL = self._fcen_grid(lgm_halos[h]) / Pnorm[h]
            idx = min(np.searchsorted(PcenL, u[h], side='right'), self.Mrvals.size - 1)
            Mr_cen[h] = self.Mrvals[idx]
        return Mr_cen

    def _nsat_grid(self, lgm):
        mhalo = 10**lgm
        Dm = (mhalo - self.M0_grid) / self.M1_grid
        return np.where(Dm < 0.0, 0.0, Dm) ** self.alpha_grid

    def _nsat_at_max(self, lgm):
        mhalo = 10**lgm
        Dm = max(0.0, (mhalo - self.M0_max) / self.M1_max)
        return Dm ** self.alpha_max

    def assign_satellite_luminosities(self, centrals, h):
        Nsat_h, lgmhalo = centrals['Nsat'][h], centrals['lgm'][h]
        PsatL = self._nsat_grid(lgmhalo) / self._nsat_at_max(lgmhalo)
        u = self.rng.rand(Nsat_h)
        idx = np.clip(np.searchsorted(PsatL, u, side='left'), 0, self.Mrvals.size - 1)
        return self.Mrvals[idx]