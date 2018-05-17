import numpy as np
import fabio
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage.interpolation import rotate
import sys

pi = np.pi
__version__ = 1.0

class SEM_FFT():
    def __init__(self):
        self.pixels_to_nm = -1
        
        self.colorbar_pad = 0.6
        #Nice Coloring:
        c = matplotlib.colors.ColorConverter().to_rgb
        custom_colors = [(0, 0, 0, 0),\
                 (0.18, 0.05, 0.05, 0.2),\
                 (0.28, 0, 0, 1),\
                 (0.4, 0.7, 0.85, 0.9),\
                 (0.45, 0, 0.75, 0),\
                 (0.6, 1, 1, 0),\
                 (0.75, 1, 0, 0),\
                 (0.92 , 0.6, 0.6, 0.6),\
                 (1  , 0.95, 0.95, 0.95)]
        cdict = {'red': [], 'green': [], 'blue': []}
        for i, item in enumerate(custom_colors):
            pos, r, g, b = item
            cdict['red'].append([pos, r, r])
            cdict['green'].append([pos, g, g])
            cdict['blue'].append([pos, b, b])
        self.cmap = matplotlib.colors.LinearSegmentedColormap('CustomMap', cdict)
        self.cmap.set_bad(color='black')
        self.sem_cmap = "gray"
    
    def get_idx(self, array, value):
        idx_sorted = np.argsort(array)
        sorted_array = np.array(array[idx_sorted])
        idx = np.searchsorted(sorted_array, value, side="left")
        if idx >= len(array):
            idx_nearest = idx_sorted[len(array)-1]
            return idx_nearest
        elif idx == 0:
            idx_nearest = idx_sorted[0]
            return idx_nearest
        else:
            if abs(value - sorted_array[idx-1]) < abs(value - sorted_array[idx]):
                idx_nearest = idx_sorted[idx-1]
                return idx_nearest
            else:
                idx_nearest = idx_sorted[idx]
            return idx_nearest
        
    def set_pixel_per_nm(self, pixel_per_nm):
        self.pixels_to_nm = 1/pixel_per_nm
        
    def set_pixel_per_size_from_tiffile(self, filepath):
        image_file = Image.open(filepath)
        info_string = image_file.tag[34118][0]
        image_pixel_size = info_string.split("Image Pixel Size",1)[-1].\
                                        split('\n')[0].\
                                        split('=')[-1].strip()
        image_pixel_size, image_pixel_unit = image_pixel_size.split()
        image_pixel_size = float(image_pixel_size)
        if image_pixel_unit != 'nm':
            if image_pixel_unit == "pm":
                image_pixel_size /= 1000
            elif image_pixel_unit == "µm":
                image_pixel_size *= 1000
            else:
                print("WARNING: Image pixel size not in nm or pm. Correct this case in code.")
                print(image_pixel_unit)
                sys.exit()
        print("Set nm/pixels conversion factor to:", image_pixel_size)
        self.pixels_to_nm = image_pixel_size

    def get_info(self, filepath):
        image_file = Image.open(filepath)
        self.info_string = image_file.tag[34118][0]
        return self.info_string

    def load_tif_file(self, filepath, set_pixelsize_from_file=True):
        self.filepath = filepath
        tiffile = fabio.open(filepath)
        self.data = tiffile.data[::-1,:].T
        self.Nx, self.Ny = self.data.shape

        if set_pixelsize_from_file:
            self.set_pixel_per_size_from_tiffile(filepath)
        self.x = np.arange(self.Nx)*self.pixels_to_nm
        self.y = np.arange(self.Ny)*self.pixels_to_nm
        
    def cut_data(self, x0=None, y0=None,\
                       x_width=None, y_height=None):
        self.y0 = y0
        self.x0 = x0
        self.x_width = x_width
        self.y_height = y_height
        
        if x_width is not None:
            if x0 is None:
                right_edge = x_width
            else:
                right_edge = x0 + x_width
        else:
            right_edge = None

        if y_height is not None:
            if y0 is None:
                upper_edge = y_height
            else:
                upper_edge = y0 + y_height
        else:
            upper_edge = None

        self.data = self.data[x0:right_edge, y0:upper_edge] 
        self.Nx, self.Ny = self.data.shape
        self.x = np.arange(self.Nx)*self.pixels_to_nm
        self.y = np.arange(self.Ny)*self.pixels_to_nm
        
    
    def annotate_point_pair(self, ax, text, xy_start, xy_end, xycoords='data',\
                text_offset=6, arrowprops = None):
        """
        Annotates two points by connecting them with an arrow. 
        The annotation text is placed near the center of the arrow.
        
        Taken from Stack Overflow.
        """

        if arrowprops is None:
            arrowprops = dict(arrowstyle= '<->')

        assert isinstance(text,str)

        xy_text = ((xy_start[0] + xy_end[0])/2. , (xy_start[1] + xy_end[1])/2.)
        arrow_vector = xy_end[0]-xy_start[0] + (xy_end[1] - xy_start[1]) * 1j
        arrow_angle = np.angle(arrow_vector)
        text_angle = arrow_angle - 0.5*np.pi

        ax.annotate(
                '', xy=xy_end, xycoords=xycoords,
                xytext=xy_start, textcoords=xycoords,
                arrowprops=arrowprops)

        label = ax.annotate(
            text, 
            xy=xy_text, 
            xycoords=xycoords,
            xytext=(text_offset * np.cos(text_angle), text_offset * np.sin(text_angle)), 
            textcoords='offset points')

        return label
    
    def plot_sem_image(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        im = self.ax.pcolormesh(self.x, self.y, self.data.T, cmap=self.sem_cmap)
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)#, orientation='horizontal')
        self.ax.set_xlabel("$ \mathit{x} \, / \, nm $")
        self.ax.set_ylabel("$ \mathit{y} \, / \, nm $")
        self.ax.set_aspect('equal')
        self.ax.set_xlim(self.x[0], self.x[-1])
        self.ax.set_ylim(self.y[0], self.y[-1])
        self.fig.tight_layout()
        plotname = self.filepath.rsplit(".",1)[0] + "_sem.png"
        self.fig.savefig(plotname)
    
    def pretty_plot(self, x0, y0, width=295, height=250, scaleBar=50):
        plotX = self.x[x0:x0+width]
        plotY = self.y[y0:y0+height]
        plotData = self.data[x0:x0+width, y0:y0+height]
        self.fig = plt.figure()
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.], )
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        im = self.ax.pcolormesh(plotX, plotY, plotData.T, cmap=self.sem_cmap)
        self.ax.set_aspect('equal')
        self.ax.set_xticklabels('')
        self.ax.set_yticklabels('')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(plotX[0], plotX[-1])
        self.ax.set_ylim(plotY[0], plotY[-1])
        self.ax.text(plotX[-1]-scaleBar/2-10, plotY[0]+15+5, '$'+str(scaleBar)+' \, nm$',\
                     horizontalalignment='center',
                     color='white')
        # verticalalignment='bottom',\
        # transform=self.ax.transAxes,\
        
        self.ax.add_patch(
            patches.Rectangle(
                (plotX[-1]-scaleBar-10, plotY[0]+10),   # (x,y)
                scaleBar,          # width
                5,          # height
                color='white'
            )
        )
        self.fig.tight_layout()
    
    def show(self):
        plt.show()
    def do_fft(self):
        self.cut_data(135, 135, 512, 512)
        fftdata = np.fft.fft2(self.data)
        absfft = np.abs(fftdata)

        fftx = np.fft.fftfreq(self.Nx, d=self.x[1]-self.x[0])*2*pi
        ffty = np.fft.fftfreq(self.Ny, d=self.y[1]-self.y[0])*2*pi

        self.fftdx = fftx[1] - fftx[0]
        self.fftdy = ffty[1] - ffty[0]

        # sort fft:
        fftnx, fftny=absfft.shape
        fftnx2 = int(round(fftnx/2.))
        fftny2 = int(round(fftny/2.))

        harrray = np.copy(absfft)
        absfft[:fftnx2,:] = harrray[fftnx2:, :]
        absfft[fftnx2:,:] = harrray[:fftnx2, :]
        harrray = np.copy(absfft)
        absfft[:,:fftny2] = harrray[:, fftny2:]
        absfft[:,fftny2:] = harrray[:, :fftny2]

        hfftx = np.copy(fftx)
        fftx[:fftnx2] = hfftx[fftnx2:]
        fftx[fftnx2:] = hfftx[:fftnx2]

        hffty = np.copy(ffty)
        ffty[:fftny2] = hffty[fftny2:]
        ffty[fftny2:] = hffty[:fftny2]

        self.fftdata_abs = absfft
        self.fftx = fftx
        self.ffty = ffty
        
        
        self.fft_centeridx_x = self.get_idx(self.fftx, 0.)
        self.fft_centeridx_y = self.get_idx(self.ffty, 0.)
        self.fftabs_maxval = np.amax(self.fftdata_abs)
        
    def plot_fft(self, k_ticker_spacing=1, plot_seperate=False):
        if plot_seperate:
            fig = plt.figure(figsize=(8/2.54, 10/2.54))
            ax = fig.add_subplot(111)
        else:
            fig = plt.figure(figsize=(16/2.54, 12/2.54))
            ax = fig.add_subplot(121)
        im = ax.pcolormesh(self.x, self.y, self.data.T, cmap=self.sem_cmap)#, cmap='gist_gray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=self.colorbar_pad)
        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
        cb.set_ticks([0, 50, 100, 150, 200, 250])
        ax.set_xlabel("$ \mathit{x} \, / \, nm $")
        ax.set_ylabel("$ \mathit{y} \, / \, nm $")
        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(self.y[0], self.y[-1])
        ax.set_aspect('equal')

        if plot_seperate:
            fig.tight_layout()
            fig.savefig(self.filepath.rsplit(".",1)[0] + "_SEM")


            fig = plt.figure(figsize=(8/2.54, 10/2.54))
            ax2 = fig.add_subplot(111)
        else:
            ax2 = fig.add_subplot(122)
        im = ax2.pcolormesh(self.fftx, self.ffty, self.fftdata_abs.T, \
                    norm=matplotlib.colors.LogNorm(1e2, self.fftabs_maxval),\
                    cmap=self.cmap)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("bottom", size="5%", pad=self.colorbar_pad)
        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
        ax2.set_xlabel("$ \mathit{k_x} \, / \, nm^{-1} $")
        ax2.set_ylabel("$ \mathit{k_y} \, / \, nm^{-1} $")
        fig.canvas.draw()
        ax2.set_aspect('equal')
        ax2.set_xlim(min(self.fftx), max(self.fftx))
        ax2.set_ylim(min(self.ffty), max(self.ffty))
        myLocator = mticker.MultipleLocator(k_ticker_spacing)
        ax2.xaxis.set_major_locator(myLocator)
        ax2.yaxis.set_major_locator(myLocator)
        fig.tight_layout()
        plotname = self.filepath.rsplit(".",1)[0] + "_fft"
        fig.savefig(plotname)
        
    def radial_integrate(self):
        y,x = np.indices(self.fftdata_abs.shape)
        r = np.sqrt((x-self.fft_centeridx_x)**2 + (y-self.fft_centeridx_y)**2)
        r = r.astype(np.int)

        tbin = np.bincount(r.ravel(), self.fftdata_abs.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin/nr
        k_values = np.arange(len(radialprofile))*self.fftdx
        filename = self.filepath.rsplit(".",1)[0] + "_fft_radial_integrated.xy"
        
        savefile = open(filename, "w")
        savefile.write("#Loaded data from: "+str(self.filepath)+"  \n")
        savefile.write("#Cutted pixels from: x,y=("+str(self.y0)+" .. "+\
                       str(self.y0+512)+")\n")
        savefile.write("#Performed FFT and radially integrated data\n")
        savefile.write("#k / nm-1\tFFT / a.u.\n")
        for ik, kval in enumerate(k_values):
            savefile.write(str(kval) + "\t" + str(radialprofile[ik]) + "\n")
        savefile.close()
        print("Saved radial integration to " + filename)
        
        fig, ax = plt.subplots()
        ax.plot(k_values, radialprofile)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("$\mathit{k} \,/\, nm^{-1}$")
        ax.set_ylabel("$FFT$")
        ax.set_xlim(min(k_values[1:]), max(k_values))
        ax.set_ylim(min(radialprofile[1:])*0.8, max(radialprofile[1:])*1.2)
        fig.tight_layout()
        plotname = self.filepath.rsplit(".",1)[0] + "_fft_radint.png"
        fig.savefig(plotname)

    def rotate_fft(self, angle, left_edge = -0.025, right_edge = 0.025):
        self.angle = angle
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.rotfftdata_abs = rotate(self.fftdata_abs, angle)
#        rotabsfft = rotabsfft[400:1000, 400:1000]
        nx, ny = self.rotfftdata_abs.shape
        pixel_center_x, pixel_center_y = 0,0
        self.rotfftx = (np.arange(nx)-nx/2)*self.fftdx
        self.rotffty = (np.arange(ny)-ny/2)*self.fftdx

        fig = plt.figure()
        ax_rot = fig.add_subplot(111)
        im = ax_rot.pcolormesh(self.rotfftx, self.rotffty, self.rotfftdata_abs.T, \
                    norm=matplotlib.colors.LogNorm(1e2, self.fftabs_maxval),\
                    cmap=self.cmap)
        divider = make_axes_locatable(ax_rot)
        cax = divider.append_axes("bottom", size="5%", pad=self.colorbar_pad)
        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
        ax_rot.set_xlabel("$ \mathit{k_x} \, / \, nm^{-1} $")
        ax_rot.set_ylabel("$ \mathit{k_y} \, / \, nm^{-1} $")
        ax_rot.set_aspect('equal')
        ax_rot.set_xlim([min(self.rotfftx), max(self.rotfftx)])
        ax_rot.set_ylim([min(self.rotffty), max(self.rotffty)])
        ax_rot.axvline(left_edge, color='black')
        ax_rot.axvline(right_edge, color='black')
        fig.tight_layout()
        plotname = self.filepath.rsplit(".",1)[0] + "_fft_rotated_by_"+\
                   str(angle)+".png"
        fig.savefig(plotname)

    def project_rotated_data(self):
        x_idx_left = self.get_idx(self.rotfftx, self.left_edge)
        x_idx_right = self.get_idx(self.rotfftx, self.right_edge)
        
        integrated_data = np.sum(self.rotfftdata_abs[x_idx_left:x_idx_right, :],\
                                 axis=0)
        

                
            
        filename = self.filepath.rsplit(".",1)[0] + "_fft_rotated_integrated.xy"
        
        savefile = open(filename, "w")
        savefile.write("#Loaded data from: "+str(self.filepath)+"  \n")
        savefile.write("#Cutted pixels from: x,y=("+str(self.y0)+\
                       " .. " + str(self.y0+512)+")\n")
        savefile.write("#Performed FFT\n")
        savefile.write("#Rotated FFT data by an angle of " + str(self.angle) +\
                       " deg\n")
        savefile.write("#Integrated data between x=(" + str(self.left_edge) +\
                        " .. " + str(self.right_edge) + ")\n")
        savefile.write("#k / nm-1\tFFT / a.u.\n")
        for ik, kval in enumerate(self.rotffty):
            int_val = integrated_data[ik]
            if np.allclose(int_val, 0.):
                continue
            savefile.write(str(kval) + "\t" + str(int_val) + "\n")
        savefile.close()
        print("Saved rotated and integrated data to " + filename)
        
        fig, ax = plt.subplots()
        ax.plot(self.rotffty, integrated_data)
        ax.set_xlabel("$ \mathit{k_y} \, / \, nm^{-1}$")
        ax.set_ylabel("$ \mathit{FFT} \, / \, a.u.$")
        ax.set_xscale('log')
        ax.set_yscale('log')
        fig.tight_layout()
        plotname = self.filepath.rsplit(".",1)[0] + "_fft_rotated_integrated.png"
        fig.savefig(plotname)
