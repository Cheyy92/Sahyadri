import matplotlib.colors as colors
import matplotlib.cm as cmx
import pylab as pl
import numpy as np

def get1dcmap(valarr,mtype='jet'):
    cmap = pl.get_cmap(mtype)
    cNorm = colors.Normalize(vmin=np.min(valarr), vmax=np.max(valarr))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    mcbar=pl.cm.ScalarMappable(cmap=cmap,norm=cNorm)
    mcbar.set_array(valarr)

    cval=scalarMap.to_rgba(valarr)

    return cval,scalarMap, mcbar

def discrete_map(valarr,mtype='jet',ncbin=20):
    '''ncbin= number of discrete bins in the colormap'''
    
    cmap = pl.get_cmap(mtype)  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(valarr.min(), valarr.max(), ncbin)
    norm = colors.BoundaryNorm(bounds, cmap.N)


    #create the color values
    cval=cmap(valarr)
    

    #colorbar ticks
    ticks_ed=np.linspace(np.min(valarr),np.max(valarr),ncbin)
    ticks=0.5*(ticks_ed[1:]+ticks_ed[:-1])
    #colorbar
    mcbar=pl.cm.ScalarMappable(cmap=cmap,norm=norm)
    mcbar.set_array(valarr)
    
    return cval, cmap, mcbar, ticks
  
    
def add_colorbar(fig,mcbar,ticks=None,title='',rotation=90,fontsize=28):
    cbar_ax1 = fig.add_axes([0.96, 0.15, 0.02, 0.7])
    
    cbar=fig.colorbar(mcbar, cax=cbar_ax1,ticks=ticks)
    
    #cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(title, rotation=rotation,fontsize=fontsize)
    
    return cbar
