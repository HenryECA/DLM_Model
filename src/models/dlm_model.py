import numpy as np
from scipy.stats import norm
from numpy.fft import rfft, rfftfreq

from models.base_model import BaseModel

class DLM(BaseModel):
    def __init__(self, F, G, V, W, ci=0.95, component_info=None):
        self.F, self.G, self.V, self.W = F, G, V, W
        self.state_mean = None
        self.state_cov = None
        self.data = []
        alpha = 1 - ci
        self.z = norm.ppf(1 - alpha/2)
        # store list of component descriptors
        self.component_info = component_info or []

    def initialize(self, mean, cov):
        self.state_mean = mean
        self.state_cov = cov

    def fit(self, data):
        self.data = list(data)
        m0 = np.zeros((self.F.shape[0], 1))
        P0 = np.eye(self.F.shape[0]) * 1.0
        self.initialize(m0, P0)
        for y in self.data:
            self.update(np.array([[y]]))

    def predict_once(self):
        a = self.F @ self.state_mean
        R = self.F @ self.state_cov @ self.F.T + self.V
        m = self.G @ a
        C = self.G @ R @ self.G.T + self.W
        return m, C

    def update(self, y):
        a = self.F @ self.state_mean
        R = self.F @ self.state_cov @ self.F.T + self.V
        Q = self.G @ R @ self.G.T + self.W
        A = R @ self.G.T @ np.linalg.inv(Q)
        self.state_mean = a + A @ (y - self.G @ a)
        self.state_cov = R - A @ Q @ A.T

    def forecast(self, horizon):
        m0, P0 = self.state_mean.copy(), self.state_cov.copy()
        preds, lowers, uppers, std_list = [], [], [], []
        m, P = m0.copy(), P0.copy()

        for _ in range(horizon):
            a = self.F @ m
            R = self.F @ P @ self.F.T + self.V
            obs_mean = self.G @ a
            obs_cov = self.G @ R @ self.G.T + self.W
            std = float(np.sqrt(obs_cov[0,0]))
            y_hat = float(obs_mean)

            preds.append(y_hat)
            lowers.append(y_hat - self.z * std)
            uppers.append(y_hat + self.z * std)
            std_list.append(std)

            m = a.copy()
            P = R.copy()

        self.state_mean, self.state_cov = m0, P0
        return preds, lowers, uppers, std_list

    def predict(self, horizon=1):
        return self.forecast(horizon)

    def reset(self):
        self.fit(self.data)

    def get_params(self):
        """
        Returns a dictionary of the DLM components and their parameters.
        Converts any NumPy arrays into native Python lists for JSON serialization.
        """
        params = {}
        for comp in self.component_info:
            name = comp['name']
            details = comp['details']
            clean_details = {}
            for k, v in details.items():
                if isinstance(v, np.ndarray):
                    clean_details[k] = v.tolist()
                else:
                    clean_details[k] = v
            params[name] = clean_details
        return params

    def get_num_params(self):
        """
        Returns the total number of parameters in the DLM model.
        """
        return sum(np.prod(v.shape) for v in [self.F, self.G, self.V, self.W])

    @classmethod
    def from_spec(cls, spec, data, ci=0.95,
                  custom_V_lvl=None, custom_V_tr=None, custom_W_obs=None, custom_V_seas=None,
                  level_factor=0.01, trend_factor=0.005, seas_factor=0.001, obs_factor=0.005):
        data = np.asarray(data)
        var_data = np.var(data)

        comps = []
        comp_info = []

        # ----- 1) LEVEL BLOCK -----
        if spec.get('level', False):
            F_lvl = np.array([[1.]])
            G_lvl = np.array([[1.]])
            V_lvl = custom_V_lvl if custom_V_lvl is not None else np.array([[np.var(np.diff(data,1)) * level_factor]])
            comps.append((F_lvl, G_lvl, V_lvl))
            comp_info.append({'name': 'level', 'details': {'V': V_lvl}})

        # ----- 2) TREND BLOCK -----
        if spec.get('trend', False):
            F_tr = np.array([[1., 1.], [0., 1.]])
            G_tr = np.array([[1., 0.]])
            V_tr = custom_V_tr if custom_V_tr is not None else np.diag([np.var(np.diff(data,2)) * trend_factor, 0.])
            comps.append((F_tr, G_tr, V_tr))
            comp_info.append({'name': 'trend', 'details': {'V': V_tr}})

        # ----- 3) SEASONAL BLOCKS -----
        seasons = spec.get('seasonal', {})
        periods = []
        if 'periods' in seasons:
            periods = seasons['periods']
        else:
            n = len(data)
            fftm = np.abs(rfft(data-np.mean(data)))**2
            power = fftm / np.sum(fftm)
            idx = np.where(power >= seasons.get('importance_th', 0))[0]
            idx = idx[idx>0]
            idx = idx[np.argsort(power[idx])[-seasons.get('n_components',1):]]
            periods = list((n/idx).astype(int))

        for p in periods:
            omega = 2*np.pi/p
            F_s = np.array([[np.cos(omega), np.sin(omega)], [-np.sin(omega), np.cos(omega)]])
            G_s = np.array([[1., 0.]])
            V_s = custom_V_seas if custom_V_seas is not None else np.eye(2)*(var_data*seas_factor)
            comps.append((F_s, G_s, V_s))
            comp_info.append({'name': f'seasonal_{p}', 'details': {'period': int(p), 'V': V_s}})

        # ----- 4) (optional) AR BLOCK -----
        ar_order = spec.get('ar', 0)
        if ar_order > 0:
            F_ar = np.zeros((ar_order, ar_order))
            G_ar = np.zeros((1, ar_order))
            V_ar = np.eye(ar_order)*(var_data*0.01)
            comps.append((F_ar, G_ar, V_ar))
            comp_info.append({'name': f'ar_{ar_order}', 'details': {'V': V_ar}})

        # Build F, V, G same as before
        total_dim = sum(c[0].shape[0] for c in comps)
        F = np.zeros((total_dim, total_dim)); V = np.zeros((total_dim, total_dim))
        G = np.hstack([c[1] for c in comps])
        offset = 0
        for Fi, Gi, Vi in comps:
            d = Fi.shape[0]
            F[offset:offset+d, offset:offset+d] = Fi
            V[offset:offset+d, offset:offset+d] = Vi
            offset += d

        W = custom_W_obs if custom_W_obs is not None else np.array([[var_data*obs_factor]])
        model = cls(F, G, V, W, ci=ci, component_info=comp_info)

        epsilon = 0.1
        P0 = np.eye(total_dim)*(var_data*epsilon)
        m0 = np.zeros((total_dim,1))
        model.initialize(m0, P0)

        model.fit(data)
        return model
