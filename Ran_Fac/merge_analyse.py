import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ran_fac_targets = np.array([1e3, 3e3, 1e4, 3e4, 1e5], dtype=int)
percentiles = np.arange(1, 100)
                        
results = np.array([np.load(f'vvf_pct_ranfac_{rf}.npy') for rf in ran_fac_targets])

#Convergence check
tol = 0.01
converged_ranfac = np.full(len(percentiles), np.nan)
for j in range(len(percentiles)):
    for i in range(1, len(ran_fac_targets)):
        remaining = results[i:, j]
        if np.all(np.abs(remaining - results[i-1,j])/results[i-1,j] < tol):
            converged_ranfac[j] = ran_fac_targets[i]
            break
            
with open('converged_ranfac.txt','w') as f:
    for p, rf in zip(percentiles, converged_ranfac):
        f.write(f"{p} {rf}\n")
        
#Plotting:
x = ran_fac_targets/ran_fac_targets.max()
plt.figure(figsize=(9,6))
for j in range(len(percentiles)):
    plt.plot(x, results[:,j], alpha=0.5, lw=0.8)
plt.xlabel(r'$Ran\_Fac/Ran\_Fac_{max}$'); plt.ylabel('$y_p$'); plt.yscale('log')
plt.savefig('vvf_convergence_all_percentiles.png', dpi=150)                   