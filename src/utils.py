import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
from scipy.stats import norm
import math
import torch
import numpy as np
from scipy.stats import norm

def seed_all(seed=42):
    """
    Set the seed for all libraries
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

def download_data():
    df = pd.read_csv("data/tourism.csv")

    # Keep columns Quarter, State Purpose and Trips (thousands)
    df = df[['Quarter', 'State', 'Purpose', 'Trips']]

    # TODO
    # filtern State by [ACT, Victoria]
    df = df[df['State'].isin(['ACT', 'Victoria'])]

    df = df.set_index('Quarter')

    columns = ["Quarter", "Total"]
    for state in df['State'].unique():
        columns.append(state)
        for purpose in df['Purpose'].unique():
            columns.append(state + " " + purpose)

    # Create the new dataframe
    df2 = pd.DataFrame(columns=columns)
    df2.set_index('Quarter', inplace=True)

    # Fill the new dataframe
    # Fill the total column as the sum of all states and purposes
    for quarter in df.index.unique():
        df2.loc[quarter, "Total"] = df.loc[quarter, 'Trips'].sum()

    hierarchy = {
        "Total": [],
    }

    # Fill the state columns
    for state in df['State'].unique():
        hierarchy["Total"].append(state)
        hierarchy[state] = []
        for quarter in df.index.unique():
            df2.loc[quarter, state] = df[df['State']==state].loc[quarter]["Trips"].sum()

    # Fill the state and purpose columns
    for state in df['State'].unique():
        for purpose in df['Purpose'].unique():
            hierarchy[state].append(state + " " + purpose)
            for quarter in df.index.unique():
                df2.loc[quarter, state + " " + purpose] = df[(df['State']==state) & (df['Purpose']==purpose)].loc[quarter]["Trips"].sum()

    # Change data format into dict of form {column: [values]}
    data_dict = {}
    for column in df2.columns:
        data_dict[column] = df2[column].values

    return data_dict, hierarchy


def rmse(targets, predictions):
    """
    Compute RMSE and its standard error.
    
    Returns
    -------
    (rmse_value, se)
        rmse_value : float
            √(mean((predictions - targets)^2))
        se : float
            standard error of the residuals = std(residuals, ddof=1) / sqrt(n)
    """
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    residuals = predictions - targets
    mse = np.mean(residuals**2)
    rmse_value = np.sqrt(mse)
    # sample standard deviation of residuals
    se = np.std(residuals, ddof=1) / np.sqrt(residuals.size)
    return rmse_value, se

def mae(targets, predictions):
    """
    Compute MAE and its standard error.
    
    Returns
    -------
    (mae_value, se)
        mae_value : float
            mean(|predictions - targets|)
        se : float
            standard error of the absolute residuals = std(|residuals|, ddof=1) / sqrt(n)
    """
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    abs_errors = np.abs(predictions - targets)
    mae_value = abs_errors.mean()
    # sample standard deviation of absolute errors
    se = np.std(abs_errors, ddof=1) / np.sqrt(abs_errors.size)
    return mae_value, se


def crps(y, y_hat, std, alpha=0.05):
    """
    Compute the (average) Continuous Ranked Probability Score (CRPS)
    under a Gaussian forecast N(y_hat, std^2), plus its standard error.

    Parameters
    ----------
    y       : array-like of true values y_i
    y_hat   : array-like of predictive means μ_i
    std     : array-like of predictive standard deviations o_i
    alpha   : total miscoverage (unused here; kept for API consistency)

    Returns
    -------
    mean_crps : float
        The average CRPS over all observations.
    se_crps   : float
        The standard error of the mean CRPS:
        std(crps_i, ddof=1) / sqrt(n)
    """
    # turn inputs into arrays
    y      = np.asarray(y)
    mu     = np.asarray(y_hat)
    sigma  = np.asarray(std)

    # guard against non-positive sigmas
    sigma  = np.where(sigma <= 0, np.nan, sigma)

    # standardized residuals
    t   = (y - mu) / sigma
    Phi = norm.cdf(t)
    phi = norm.pdf(t)

    # per-observation CRPS
    crps_i = sigma * ( t * (2*Phi - 1) + 2*phi - 1/np.sqrt(np.pi) )

    # compute mean and standard error
    mean_crps = np.nanmean(crps_i)
    # sample standard deviation (ddof=1) of the valid entries
    valid = ~np.isnan(crps_i)
    n     = np.sum(valid)
    if n > 1:
        se_crps = np.nanstd(crps_i, ddof=1) / np.sqrt(n)
    else:
        se_crps = np.nan

    return mean_crps, se_crps


def nll(y, y_hat, std, alpha=0.05):
    """
    Compute the (average) Negative Log-Likelihood under a Gaussian
    whose mean is y_hat and std is std, plus its standard error.

    Returns
    -------
    mean_nll : float
        -mean_i [ log p(y_i | μ_i, σ_i) ]
    se_nll   : float
        standard error of the per-sample NLLs:
        std(nll_i, ddof=1) / sqrt(n)
    """
    y     = np.asarray(y)
    mu    = np.asarray(y_hat)
    sigma = np.asarray(std)

    # guard against non-positive sigmas
    sigma = np.where(sigma <= 0, np.nan, sigma)

    # per-sample negative log-likelihoods
    ll_i    = norm.logpdf(y, loc=mu, scale=sigma)
    nll_i   = -ll_i

    # compute mean and standard error
    mean_nll = np.nanmean(nll_i)
    valid    = ~np.isnan(nll_i)
    n        = valid.sum()
    if n > 1:
        se_nll = np.nanstd(nll_i, ddof=1) / np.sqrt(n)
    else:
        se_nll = np.nan

    return mean_nll, se_nll


def calibration(y, y_hat, std, alpha=0.05):
    y      = np.asarray(y)
    mu     = np.asarray(y_hat)
    sigma  = np.asarray(std)
    sigma  = np.where(sigma <= 0, np.nan, sigma)
    z      = norm.ppf(1 - alpha/2)
    lower  = mu - z * sigma
    upper  = mu + z * sigma
    inside = (y >= lower) & (y <= upper)
    return np.nanmean(inside)

def expected_calibration_error(
    y,
    y_hat,
    std,
    alphas=None,
    n_bins=10
):
    """
    Compute the Expected Calibration Error (ECE) across multiple central-interval levels,
    plus its standard error.

    ECE = mean_j | calibration(alpha_j) - (1 - alpha_j) |
    SE  = std_j( errors ) / sqrt( number_of_alphas )

    Parameters
    ----------
    y       : array-like of true values y_i
    y_hat   : array-like of predictive means μ_i
    std     : array-like of predictive standard deviations σ_i
    alphas  : array-like of miscoverage levels α_j (total miscoverage);
              if None, defaults to n_bins evenly spaced in (0,1)
    n_bins  : number of α-levels to use when alphas is None

    Returns
    -------
    ece     : float
              mean absolute deviation between empirical and nominal coverage.
    se_ece  : float
              standard error of those binwise errors.
    """
    if alphas is None:
        # avoid exactly 0 or 1
        alphas = np.linspace(0.01, 0.99, n_bins)

    errors = []
    for α in alphas:
        cov = calibration(y, y_hat, std, alpha=α)
        errors.append(abs(cov - (1 - α)))

    errors = np.array(errors)
    ece    = np.nanmean(errors)
    # sample std dev of the errors
    m      = np.sum(~np.isnan(errors))
    if m > 1:
        se_ece = np.nanstd(errors, ddof=1) / np.sqrt(m)
    else:
        se_ece = np.nan

    return ece, se_ece



def run_metrics(y, mean, std, alpha):
    """
    Run all metrics on the given data
    """

    results = {
        'RMSE': rmse(y, mean),
        'MAE': mae(y, mean),
        'CRPS': crps(y, mean, std, alpha),
        'NLL': nll(y, mean, std, alpha),
        'ECE': expected_calibration_error(y, mean, std),
    }

    metrics = {key: value[0] for key, value in results.items()}
    metrics_se = {key: value[1] for key, value in results.items()}

    return metrics, metrics_se

def save_data_point(data, x, h):
    """
    Save x in the data array depending on the horizon
    """
    # data is a list of lists. Each inner list corresponds with the possible values for that prediction

    if len(data) !=  h:
        raise ValueError(f"Data length {len(data)} does not match horizon {h}")

    for i, d in enumerate(x):
        data[i].append(d)

    return data


if __name__ == '__main__':
    download_data()
