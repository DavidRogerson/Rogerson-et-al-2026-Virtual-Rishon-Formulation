import matplotlib.pyplot as plt
import numpy as np
import h5py
from tenpy.tools.hdf5_io import Hdf5Loader, Hdf5Saver
from scipy.optimize import curve_fit


if __name__ == "__main__":
    # Example data
    with h5py.File("results_simulation.h5", 'r') as f:
        y=measurements = Hdf5Loader(f['measurements']['entropy']).load()
        params = Hdf5Loader(f['simulation_parameters']['model_params']).load()
    y = y[-1, :]  # Simulated entropy data
    bc = params['bc_x']
    Nf = params['Nf']
    Nr = params.get('Nr',1)
    L = params['L']
    m = params['m']
    a = params['a']
    g = params['g']
    theta = params['theta']

    r = np.arange(1,len(y)+1)/(Nf+Nr)
    start = 3
    stop = L-2
    subset = np.array([i*(Nf+Nr)-1 for i in range(start,stop)])
    y_fit = y[subset]  # For 
    r_fit = np.arange(start, stop,1)
    if bc == 'open':
        def fit_func(r,c, s0):
            return c/6*np.log(2*L/(np.pi)*np.sin(np.pi/(L)*r)) + s0
    elif bc == 'periodic':
        def fit_func(r,c, s0):
            return c/3*np.log(L/np.pi * np.sin(np.pi/(L)*r)) + s0
    else:
        raise ValueError("Boundary condition not recognized. Use 'open' or 'periodic'.")
    # Fit the data
      # Assuming r is a 1D array corresponding to the entropy values
    print(r_fit, L)
    popt, pcov = curve_fit(fit_func, r_fit, y_fit, p0=[1, 0])  # Initial guess for parameters c and s0
    c, s0 = popt
    plt.figure(figsize=(10, 6))
    plt.plot(r, y, marker='o', linestyle='', color='b', label='Entropy')
    plt.plot(r, fit_func(r, *popt), color='r', label=f'Fit: $c={c:.3f} \\pm {np.sqrt(pcov[0,0]):.3f}, s0={s0:.3f} \\pm {np.sqrt(pcov[1,1]):.3f}$')
    plt.plot(r_fit, y_fit, '.', color='k')

    plt.title(f'Entropy vs. r: m={m}, a={a}, g={g}, Nf={Nf}, bc={bc}, theta={theta}')
    plt.xlabel('r')
    plt.ylabel('Entropy')
    plt.grid(True)
    plt.legend()
    plt.savefig("entropy_plot_per_unit.png")

    with h5py.File("results_central_charge_per_unit.h5", 'w') as f:
        saver = Hdf5Saver(f)
        saver.save(r, '/all_cuts')
        saver.save(y, '/all_entropies')
        saver.save(r_fit, '/fitted_cuts')
        saver.save(y_fit, '/fitted_entropies')
        saver.save(fit_func(r, *popt), '/model_preditions')
        saver.save('lambda r, c, s0: c/6*np.log(np.sin(np.pi/(L)*r)) + s0' if bc=='open' else 'lambda r, c, s0: c/3*np.log(np.sin(np.pi/(L)*r)) + s0', '/model')
        saver.save(popt, '/params')
        saver.save(pcov, '/cov')