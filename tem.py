import lmfit, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import javabridge
import bioformats

class TEM():
  def __init__(self):
    self.counts = None
    self.errors = None
    self.bin_width = None
    self.bins = None
    self.integrated_sum = None
    
    self.fig = None

    self.p = None
    self.sem_cmap = "gray"
    self.initialize_data()
  
  def initialize_data(self):
    self.Nbins = 10
  
  def init_vm(self):
    javabridge.start_vm(class_path=bioformats.JARS)

    #remove annoying logs
    myloglevel="ERROR"  # user string argument for logLevel.
    rootLoggerName = javabridge.get_static_field("org/slf4j/Logger","ROOT_LOGGER_NAME", "Ljava/lang/String;")
    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory","getLogger", "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level",myloglevel, "Lch/qos/logback/classic/Level;")
    javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

  def kill_vm(self):
    javabridge.kill_vm()

  def load_tif_file(self, tiffile, init_vm=True, kill_vm_on_end=True):
    # get metadata
    metadata = os.popen(
      os.path.dirname(os.path.realpath(__file__))+'/showinf ' + tiffile +\
      ' -nopix'
    ).read()
    
    #Acquisition

    pixel_size_x = metadata.split("Nanometers per pixel (X)",1)[-1]\
                  .split('\n')[0]\
                  .split(':')[-1].strip()
    pixel_size_y = metadata.split("Nanometers per pixel (Y)",1)[-1]\
                  .split('\n')[0]\
                  .split(':')[-1].strip()
    if(pixel_size_x != pixel_size_y):
      print("WARNING. PIXEL SIXE X != PIXEL SIZE Y. UNEXPECTED")
      print('Pixel Size X:', pixel_size_x)
      print('Pixel Size Y:', pixel_size_y)
      print('Continuing with pixel size X only')
    
    self.nm_per_pixel = float(pixel_size_x)
    
    # get image
    if init_vm:
      self.init_vm()

    #properly select image reader to load data
    #see: https://github.com/CellProfiler/python-bioformats/issues/23
    image_reader = bioformats.formatreader.make_image_reader_class()()
    image_reader.allowOpenToCheckType(True)
    image_reader.setId(tiffile)
    wrapper = bioformats.formatreader.ImageReader(path=tiffile, perform_init=False)
    wrapper.rdr = image_reader
    self.data = wrapper.read()[::-1,:].T
    if kill_vm_on_end:
      javabridge.kill_vm()

    self.Nx, self.Ny = self.data.shape
    
    self.x = np.arange(self.Nx)*self.nm_per_pixel
    self.y = np.arange(self.Ny)*self.nm_per_pixel
  
  def pretty_plot(self, x0, y0, width=295, height=250, scaleBar=50):
    plotX = self.x[x0:x0+width]
    plotY = self.y[y0:y0+height]
    plotData = self.data[x0:x0+width, y0:y0+height]
    self.fig = plt.figure()
    self.ax = self.fig.add_subplot(111)
    self.ax.pcolormesh(plotX, plotY, plotData.T, cmap=self.sem_cmap)
    self.ax.set_aspect('equal')
    self.ax.set_xticklabels('')
    self.ax.set_yticklabels('')
    self.ax.set_xticks([])
    self.ax.set_yticks([])
    self.ax.set_xlim(plotX[0], plotX[-1])
    self.ax.set_ylim(plotY[0], plotY[-1])

    r = self.fig.canvas.get_renderer()
    
    t = self.ax.text(
      0,0,str(scaleBar)+' nm',\
      horizontalalignment='center',
      color='white')
    bb = t.get_window_extent(renderer=r)
    textWidth = bb.width*self.nm_per_pixel
    textHeight = bb.height*self.nm_per_pixel
    # print(textWidth)
    
    t.remove()
    offset = 30*self.nm_per_pixel
    self.ax.text(
      plotX[-1] - offset - textWidth,\
      plotY[0]+height*2/20*self.nm_per_pixel+5*self.nm_per_pixel,\
      '$'+str(scaleBar)+' \, nm$',\
      horizontalalignment='center',
      color='white')
  
    self.ax.add_patch(
      patches.Rectangle(
        (plotX[-1] - offset - textWidth - scaleBar*1/2.,\
        plotY[0]+height*1/20*self.nm_per_pixel),   # (x,y)
        scaleBar,          # width
        height*1/40*self.nm_per_pixel,          # height
        color='white'
      )
    )
    self.fig.tight_layout()
      
  def load(self, distances):
    self.L = distances

  def prepare_length_histogram(self):
    self.make_count_histogram(self.L)
    self.xlabel = "$\mathit{L} \, / \, nm$"
    
  def prepare_aspect_histogram(self):
    print("Prepare Aspect Histogram not defined.")
  
  def make_count_histogram(self, lengths):
    self.raw_data = lengths
    self.counts, self.bin_edges= np.histogram(lengths, bins=self.Nbins)
    self.errors = np.sqrt(self.counts)
    self.bin_width = self.bin_edges[1] - self.bin_edges[0]
    self.bins = self.bin_edges[:-1] + self.bin_width/2.
    self.integrated_sum = sum(self.counts)*self.bin_width
  
  #Lognormal Fitting
  def init_lognormal(self, mu=1, logstd=0.1):
    self.p = lmfit.Parameters()
    self.p.add("N", 1, min=0)
    self.p.add("logmu", mu, min=0.)
    self.p.add("logstd", logstd, min=0)
    
  def fit_lognormal(self, mu0=10, std0=0.1):
    self.init_lognormal(mu0, std0)
    print("Fitting data to lognormal function...")
    
    valid_values = self.counts > 0
    self.fitresults = lmfit.minimize(self.lognormal_residual, self.p,\
            args=(self.bins[valid_values],\
                self.counts[valid_values],\
                self.errors[valid_values]))
    print(lmfit.fit_report(self.fitresults))
    self.p_result = self.fitresults.params
  
  def plot_histogram(self, savename=None,\
            show=True):
    if self.fig is None:
      self.fig = plt.figure()
    self.ax = self.fig.add_subplot(111)

    self.ax.errorbar(self.bins, self.counts, self.errors,\
          markersize=0, capsize=2, elinewidth=1, linestyle='None', color='darkgreen')
    self.ax.hist(self.raw_data, bins=self.Nbins, alpha=0.7, facecolor='green')
    
    x_for_fit_display = np.linspace(self.bins[0]*0.9, self.bins[-1]*1.1, 100)
    
    self.ax.set_xlabel(self.xlabel)
    self.ax.set_ylabel("$counts$")
#    self.ax.tick_params(self.axis='both', pad=6, width=1, length=2)
    self.ax.set_xlim([self.bins[0]-self.bin_width, self.bins[-1]+self.bin_width])
    self.ax.legend(loc='best')
    self.fig.tight_layout()
    if savename is not None:
      self.save_plot(savename)
    if show:
      plt.show()

  def plot_lognormal(self, savename=None,\
            mu_unit="nm", mu_sf=1, sigma_unit="\%", sigma_sf=100,\
            show=True):
    if self.fig is None:
      self.fig = plt.figure()
    self.ax = self.fig.add_subplot(111)

    self.ax.errorbar(self.bins, self.counts, self.errors,\
          markersize=0, capsize=2, elinewidth=1, linestyle='None', color='darkgreen')
    self.ax.hist(self.raw_data, bins=self.Nbins, alpha=0.7, facecolor='green')
    
    x_for_fit_display = np.linspace(self.bins[0]*0.9, self.bins[-1]*1.1, 100)
    self.ax.plot(x_for_fit_display,\
        self.lognormal(self.p_result, x_for_fit_display),\
        color='black', lw=1, marker='None')
    self.ax.plot([], [], marker='None', ls='None', label=\
             "$\mu_{log}\,=\,"+\
               "{:.1f}".format(self.p_result["logmu"].value*mu_sf)+\
               "\,"+mu_unit+"$\n"+
             "$\sigma_{log}\,=\,"+\
               "{:.1f}".format(self.p_result["logstd"].value*sigma_sf)+\
               "\,"+sigma_unit+"$")

    self.ax.set_xlabel(self.xlabel)
    self.ax.set_ylabel("$counts$")
#    self.ax.tick_params(self.axis='both', pad=6, width=1, length=2)
    self.ax.set_xlim([self.bins[0]-self.bin_width, self.bins[-1]+self.bin_width])
    self.ax.legend(loc='best')
    self.fig.tight_layout()
    if savename is not None:
      self.save_plot(savename)
    if show:
      plt.show()
  def save_plot(self, savename, dpi=None):
    if dpi is not None:
      self.fig.savefig(savename, dpi=dpi)
    else:
      self.fig.savefig(savename)
    print("Saved plot to " + savename)
    
  def show(self):
    plt.show()
  #Gaussian Fitting
  def init_gaussian(self, mu=1, std=0.1):
    self.p = lmfit.Parameters()
    self.p.add("N", 1, min=0)
    self.p.add("mu", mu, min=0.)
    self.p.add("std", std, min=0)
    
  def fit_gaussian(self, mu0=1, std0=0.1):
    self.init_gaussian(mu0, std0)
    print("Fitting data to gaussian function...")
    valid_values = self.counts > 0
    self.fitresults = lmfit.minimize(self.gaussian_residual, self.p,\
            args=(self.bins[valid_values],\
                self.counts[valid_values],\
                self.errors[valid_values]))
    print(lmfit.fit_report(self.fitresults))
    self.p_result = self.fitresults.params
    
  def plot_gaussian(self, savename="GaussianFit.png",\
            mu_unit="nm", mu_sf=1, sigma_unit="nm", sigma_sf=1):
    fig, self.ax = plt.subplots()
    self.ax.errorbar(self.bins, self.counts, self.errors,\
          markersize=0, capsize=2, elinewidth=1, linestyle='None', color='darkgreen')
    self.ax.hist(self.raw_data, bins=self.Nbins, alpha=0.7, facecolor='green')
    
    x_for_fit_display = np.linspace(self.bins[0]*0.9, self.bins[-1]*1.1, 100)
    self.ax.plot(x_for_fit_display,\
        self.gaussian(self.p_result, x_for_fit_display),\
        color='black', lw=1, marker='None',\
        label= "$\mu\,=\,"+\
               "{:.1f}".format(self.p_result["mu"].value*mu_sf)+\
               "\,"+mu_unit+"$\n"+
             "$\sigma\,=\,"+\
               "{:.1f}".format(self.p_result["std"].value*sigma_sf)+\
               "\,"+sigma_unit+"$")

    self.ax.set_xlabel(self.xlabel)
    self.ax.set_ylabel("$counts$")
    self.ax.set_xlim([self.bins[0]-self.bin_width, self.bins[-1]+self.bin_width])
    self.ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(savename)
    print("Saved plot to " + savename)
    plt.show()
    
  def export_fit_result(self, export_filename="FitResult.dat"):
    savefile = open(export_filename, "w")
    savefile.write("#"+str(lmfit.fit_report(self.fitresults)).replace("\n","\n#"))
    
    savefile.write("\n\n#bin_center\tcount\terror\n")
    for ib, binval in enumerate(self.bins):
      savefile.write(str(binval)+"\t"+str(self.counts[ib])+"\t"+\
               str(self.errors[ib])+"\n")
    savefile.close()
    print("Saved fit result to "+ export_filename)
  # Functions
  def lognormal(self, p, x):
    N = p["N"].value
    mu = p["logmu"].value
    logstd = p["logstd"].value
    return N/(np.sqrt(2*np.pi)*logstd*x) *\
               np.exp(- ((np.log(x/mu))/(logstd))**2 / 2.)

  def gaussian(self, p, x):
    N = p["N"].value
    mu = p["mu"].value
    std = p["std"].value
    return N/(np.sqrt(2*np.pi)*std) * np.exp(- ((x-mu)/std)**2 / 2.)
  
  def lognormal_residual(self, p, x, y, sy):
    return (self.lognormal(p, x) - y)/sy

  def gaussian_residual(self, p, x, y, sy):
    return (self.gaussian(p, x) - y)/sy
  
  def load_csv(self, csvfile, delimiter=None):
    csvdata = np.genfromtxt(csvfile, delimiter=delimiter)
    L = csvdata[:,-1]
    self.load(L)
    return L

  def load_xls(self, xlsfile, lengthidx=-1):
    xlsdata = open(xlsfile, "r", encoding='utf-8', errors='ignore')
    lengths = []
    for line in xlsdata:
      if ',' in line:
        split_line = line.strip().split(',')
      else:
        split_line = line.strip().split()
      if len(split_line) == 0:
        continue
      try:
        first_entry_number = float(split_line[0])
      except ValueError:
        continue
      length_value = float(split_line[lengthidx].replace(",","."))
      lengths.append(length_value)
    xlsdata.close()
    lengths = np.asarray(lengths)
    self.load(lengths)
    return lengths
  
  def merge_arrays(self, a, b):
    return np.concatenate([a,b])
    
class TEMCubes(TEM):
  def __init__(self):
    super().__init__()
  
  def load(self, alternating_edge_lengths):
    self.L = np.asarray(alternating_edge_lengths)
    self.a = self.L[::2]
    self.b = self.L[1::2]
    self.aspect = self.b/self.a

  def prepare_length_histogram(self):
    self.make_count_histogram(self.L)
    self.xlabel = "$\mathit{a} \, / \, nm$"
  
  def prepare_aspect_histogram(self):
    self.make_count_histogram(self.aspect)
    self.xlabel = "$\mathit{b} \, / \, \mathit{a}$"
    
class TEMSpheres(TEM):
  def __init__(self):
    super().__init__()
  
  def load(self, diameters):
    self.L = diameters

  def prepare_length_histogram(self):
    self.make_count_histogram(self.L)
    self.xlabel = "$\mathit{d} \, / \, nm$"

class TEMSpindles(TEM):
  def __init__(self):
    super().__init__()
  
  def load(self, alternating_spindle_lengths):
    self.L = np.asarray(alternating_spindle_lengths)
    self.a = self.L[::2]
    self.b = self.L[1::2]
    self.aspect = self.a/self.b

  def prepare_length_histogram_a(self):
    self.make_count_histogram(self.a)
    self.xlabel = "$\mathit{a} \, / \, nm$"

  def prepare_length_histogram_b(self):
    self.make_count_histogram(self.b)
    self.xlabel = "$\mathit{b} \, / \, nm$"
  
  def prepare_aspect_histogram(self):
    self.make_count_histogram(self.aspect)
    self.xlabel = "$\mathit{a} \, / \, \mathit{b}$"
