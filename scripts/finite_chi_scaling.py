import numpy as np
import h5py
from tenpy.tools.hdf5_io import Hdf5Loader
import pandas as pd
import glob
import re
from uncertainties import ufloat
import matplotlib.pyplot as plt
import signac


def load_data():
    df = {'entropy': [], 'energy':[], 'max_trunc_err':[], 'chi':[], 'Delta_E':[], 'chi_max':[], 'time':[], 'memory_in_MB':[], 'sweeps':[]}
    pattern = r"Current memory usage\s+([\d\.]+)(?=MB)"
    indices = []
    for res_file in glob.glob('results_simulation_chi_max_*.h5'):
        match = re.search(r"results_simulation_chi_max_(\d+)\.h5", res_file)
        if match:
            chi_max = int(match.group(1))
            with h5py.File(res_file, 'r') as f:
                loader = Hdf5Loader(f)
                Entropy = loader.load('measurements/entropy')
                energy = loader.load('energy')
                chi = np.max(loader.load('measurements/bond_dimension')[-1, :])
                trunc_err = loader.load('sweep_stats/max_trunc_err')
                time = loader.load('sweep_stats/time')[-1]
                sweeps = loader.load('sweep_stats/sweep')[-1]
                Delta_E = loader.load('sweep_stats/Delta_E')
            df['entropy'].append(Entropy[-1, Entropy.shape[1]//2])

            memory= []
            with open(f"results_simulation_chi_max_{chi_max}.log", 'r') as f:
                for line in f:
                    match_mem = re.search(pattern, line)
                    if match_mem:
                        # Convert the captured string to a float and append to list
                        memory.append(float(match_mem.group(1)))
            
            df['energy'].append(energy)
            df['max_trunc_err'].append(trunc_err[-1])
            df['Delta_E'].append(Delta_E[-1])
            df['chi_max'].append(chi_max)
            df['chi'].append(chi)
            df['time'].append(time)
            df['sweeps'].append(sweeps)
            df['memory_in_MB'].append(memory[-1])
    df = pd.DataFrame(df)

    df = df.sort_values('chi_max').reset_index()
    df['time_cum'] = df['time'].cumsum()
    df.to_csv('finite_chi_scaling_data_measurements.csv')
    return df

def linear_fit_interpolation(x,y,x_lab, y_lab):
    plt.title(f"0 Interpolation: {y_lab} vs {x_lab}")
    plt.plot(x,y, 'x')
    try:
        test=y[2]
        para, cov = np.polyfit(x[-3:],y[-3:], 1, cov=True)
        p = ufloat(para[1], np.sqrt(cov[1,1]))
        x_fit = np.linspace(0, np.max(x),5)
        plt.plot(x_fit, x_fit*para[0] + para[1], label=f"{y_lab}({x_lab} $= 0$) $= {p:.1u}$")
    except (np.linalg.LinAlgError, KeyError):
        p =  ufloat(float('Nan'), float('Nan'))

    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.legend()
    return p, plt.gca()


def linear_fit_slope(x,y,x_lab, y_lab):
    plt.title(f"Slope: {y_lab} vs {x_lab}")
    plt.plot(x,y, 'x')
    try:
        test=y[2]
        para, cov = np.polyfit(x[-3:],y[-3:], 1, cov=True)
        p = ufloat(para[0], np.sqrt(cov[0,0]))
        span = np.max(x) - np.min(x)
        x_fit = np.linspace(np.min(x)- 0.1*span, np.max(x)+0.1*span,5)
        plt.plot(x_fit, x_fit*para[0] + para[1], label=f"$d$ {y_lab} $/ d$ {x_lab} $ = {p:.1u}$")
    except (np.linalg.LinAlgError, KeyError):
        p =  ufloat(float('Nan'), float('Nan'))

    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.legend()
    return p, plt.gca()


def perform_fits(df):
    interpolate = [{'x':df['max_trunc_err'],
               'y':df['energy'],
               'x_lab':"$\epsilon$",
               'y_lab':"$E$"
               },
               {'x':1/df['chi'],
               'y':df['energy'],
               'x_lab':"$\chi^{-1}$",
               'y_lab':"$E$"
               },
               {'x':df['max_trunc_err'],
               'y':df['entropy'],
               'x_lab':"$\epsilon$",
               'y_lab':"$S$"
               },

               {'x':1/df['chi'],
               'y':df['entropy'],
               'x_lab':"$\chi^{-1}$",
               'y_lab':"$S$"
               },
               ]

    slopes = [

        {'x':np.log(df['chi']),
        'y':np.log(df['memory_in_MB']),
        'x_lab':"$log \chi$",
        'y_lab':"$ log RAM$ in MB"
        },
        {'x':np.log(df['chi']),
        'y':np.log(df['time']/df['sweeps']),
        'x_lab':"$log \chi$",
        'y_lab':"$\log t$ per sweep in s"
        },
    ]
    fit_extracted_names = ['E_epsilon', 'E_inv_chi', 'S_epsilon', 'S_inv_chi']
    fit_extracted = {**{key:[] for key in fit_extracted_names}, **{'Delta_' + key:[] for key in fit_extracted_names}}
    for fit_name, params in zip(fit_extracted_names, interpolate):
        fit, p = linear_fit_interpolation(**params)
        p.get_figure().savefig("finite_chi_scaling_zero_interpolation_"+fit_name+".png")
        plt.close()
        fit_extracted[fit_name].append(fit.n)
        fit_extracted['Delta_'+fit_name].append(fit.s)

    fit_extracted_names = ['Mem_power_chi', 'Time_power_chi']
    fit_extracted = {**fit_extracted, **{key:[] for key in fit_extracted_names}, **{'Delta_' + key:[] for key in fit_extracted_names}}
    for fit_name, params in zip(fit_extracted_names, slopes):
        fit, p = linear_fit_slope(**params)
        p.get_figure().savefig("finite_chi_scaling_power_law_"+fit_name+".png")
        plt.close()
        fit_extracted[fit_name].append(fit.n)
        fit_extracted['Delta_'+fit_name].append(fit.s)

    df_fits = pd.DataFrame(fit_extracted)
    df_fits.to_csv('finite_chi_scaling_data_fits.csv')


if __name__ == "__main__":
    df = load_data()
    perform_fits(df)

