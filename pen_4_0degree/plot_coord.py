import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.stats import gaussian_kde
import os

arr = np.arange(-0.6, 2.01, 0.2)

parent_dir='pen_4_0degree/'
found_files = []
peak_xs, peak_ys = [], []

all_x, all_y = [], []
for param in arr:
    fname =parent_dir+f'G_x_3_y_1_{param:.1f}/poly_struct.csv'
 

    try:
        data = np.loadtxt(fname, delimiter=',')
        x, y = data[:, 0], data[:, 1]
        all_x.append(x)
        all_y.append(y)
        found_files.append(fname)
    except Exception as e:
        print(f"Missing: {fname}")

 

all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)
max_all=np.max([np.abs(all_x.max()),np.abs(all_y.max()),np.abs(all_x.min()),np.abs(all_y.min()) ])
min_all=-max_all

global_xlim = (min_all-1, max_all+1)
global_ylim = (min_all-1, max_all+1)

 
n = len(arr)
cols = int(np.ceil(np.sqrt(n)))
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), squeeze=False)
min_v, max_v = 0, 0.08

 
for idx, param in enumerate(arr):
    row, col = divmod(idx, cols)
    ax = axes[row][col]
    fname = parent_dir+f'G_x_3_y_1_{param:.1f}/coord_data.csv'
    fname_poly =parent_dir+f'G_x_3_y_1_{param:.1f}/poly_struct.csv'

    if not os.path.exists(fname):
        ax.axis('off')
        continue
    data = np.loadtxt(fname, delimiter=',')
    poly_ = np.loadtxt(fname_poly, delimiter=',')
    x, y = data[:, 0], data[:, 1]

    poly_x=poly_[:,0]
    poly_y=poly_[:,1]
    poly_x = np.append(poly_x, poly_x[0])
    poly_y = np.append(poly_y, poly_y[0])

    nbins = 400
    xi = np.linspace(*global_xlim, nbins)
    yi = np.linspace(*global_ylim, nbins)
    xi, yi = np.meshgrid(xi, yi)
    kde = gaussian_kde(np.vstack([x, y]), bw_method=0.07)
    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
    zi = zi / np.max(zi)
    
    ax.pcolormesh(xi, yi, np.ma.masked_where(zi < 0, zi), shading='auto', cmap='viridis', vmin=min_v, vmax=max_v)
    ax.plot(poly_x,poly_y,c='k')
    ax.set_xlim(global_xlim)
    ax.set_ylim(global_ylim)
    ax.set_title(f"param={param:.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
 
for i in range(n, rows * cols):
    row, col = divmod(i, cols)
    axes[row][col].axis('off')


fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.colorbar(
    plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min_v, vmax=max_v)),
    ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label='Density'
)
plt.savefig( parent_dir+"dataout.png", dpi=500)