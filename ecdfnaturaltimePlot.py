import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# 1. Define the Distributions Dictionary
# Mapping Name -> (Scipy Object, Parameter Names)
DISTRIBUTIONS = {
    "Weibull": (st.weibull_min, ["c", "loc", "scale"]),
    "Lognormal": (st.lognorm, ["s", "loc", "scale"]),
    "Gamma": (st.gamma, ["a", "loc", "scale"]),
    "Exponential": (st.expon, ["loc", "scale"]),
    "Normal": (st.norm, ["loc", "scale"])
}

def load_and_clean_data(filename):
    try:
        df = pd.read_excel(filename)
        # clean NaNs
        data = df['Cycle_peak'].dropna()
        # Sort data (Critical for ECDF)
        data = data.sort_values()
        return data
    except FileNotFoundError:
        print("File not found. Generating dummy data.")
        # Generating dummy Weibull-like data
        return pd.Series(st.weibull_min.rvs(1.5, loc=0, scale=500, size=200)).sort_values()

def calculate_ecdf(data):
    """
    Calculates the x and y values for the Empirical CDF
    """
    n = len(data)
    x = np.sort(data)
    # y values are 1/n, 2/n, ..., n/n
    y = np.arange(1, n + 1) / n
    return x, y

def fit_and_plot(data):
    # 1. Setup the plot
    plt.figure(figsize=(12, 8))
    
    # 2. Plot the Empirical Data (The "Truth")
    x_emp, y_emp = calculate_ecdf(data)
    # 'step' plot is statistically the most accurate way to represent ECDF
    plt.step(x_emp, y_emp, label='Empirical Data (ECDF)', color='black', linewidth=2, where='post')

    # Create an x-range for plotting smooth theoretical lines
    x_space = np.linspace(min(data), max(data), 500)
    
    results = []

    print(f"{'Distribution':<15} | {'AIC':<10} | {'KS Stat':<10}")
    print("-" * 40)

    # 3. Loop through distributions and fit
    for name, (dist, param_names) in DISTRIBUTIONS.items():
        try:
            # --- Your Fitting Logic ---
            # fit(data, floc=0) forces the location to 0 (useful for cycle peaks which can't be negative)
            params = dist.fit(data, floc=0) 

            # Compute Statistics
            loglik = np.sum(dist.logpdf(data, *params))
            k = len(params)
            aic = 2*k - 2*loglik
            ks_stat, ks_pval = st.kstest(data, dist.cdf, args=params)
            
            # --- Plotting the Theoretical CDF ---
            # Calculate the CDF for the smooth x_space using fitted params
            cdf_fitted = dist.cdf(x_space, *params)
            
            # Plot line
            plt.plot(x_space, cdf_fitted, label=f'{name} (AIC: {aic:.0f})', alpha=0.7, linestyle='--')

            results.append({
                "Distribution": name,
                "AIC": aic,
                "KS": ks_stat
            })
            
            print(f"{name:<15} | {aic:<10.2f} | {ks_stat:<10.4f}")

        except Exception as e:
            print(f"Could not fit {name}: {e}")

    # 4. Finalize Plot Aesthetics
    plt.title('ECDF vs Fitted Theoretical Distributions', fontsize=16)
    plt.xlabel('Natural Times', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.legend(title="Distributions (sorted by fit)", loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.05)
    
    # Identify Best Fit based on AIC (lowest is best)
    if results:
        best_dist = min(results, key=lambda x: x['AIC'])
        plt.text(0.05, 0.95, f"Best Fit: {best_dist['Distribution']}", 
                 transform=plt.gca().transAxes, fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('ecdf_vs_distributions.png')
    plt.show()

# --- Execution ---

file_path = "/content/Taiwan Datasheet 3.5a.xlsx"
data = load_and_clean_data(file_path)
fit_and_plot(data)
