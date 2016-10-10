import lmfit
import numpy as np
import matplotlib.pyplot as plt

class TEMEvaluator():
    def __init__(self):
        self.counts = None
        self.errors = None
        self.bin_width = None
        self.bins = None
        self.integrated_sum = None
        
        self.p = None
        self.initialize_data()
    
    def initialize_data(self):
        self.Nbins = 10
        
    def load_spheres(self, diameters):
        self.sphere_diameters = diameters
        self.make_count_histogram(self.sphere_diameters)
        self.xlabel = "$\mathit{d} \, / \, nm$"
        
    def load_cubes(self, alternating_edge_lengths):
        self.cube_L = np.asarray(alternating_edge_lengths)
        self.cube_a = self.cube_L[::2]
        self.cube_b = self.cube_L[1::2]
        self.cube_aspect = self.cube_b/self.cube_a
    
    def prepare_cube_length_histogram(self):
        self.make_count_histogram(self.cube_L)
        self.xlabel = "$\mathit{a} \, / \, nm$"
    
    def prepare_cube_aspect_histogram(self):
        self.make_count_histogram(self.cube_aspect)
        self.xlabel = "$\mathit{b} \, / \, \mathit{a}$"
    
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
        self.fitresults = lmfit.minimize(self.lognormal_residual, self.p,\
                        args=(self.bins, self.counts, self.errors))
        print(lmfit.fit_report(self.fitresults))
        self.p_result = self.fitresults.params
        
    def plot_lognormal(self, savename="LognormalFit.png",\
                        mu_unit="nm", mu_sf=1, sigma_unit="\%", sigma_sf=100):
        fig, ax = plt.subplots()
        ax.errorbar(self.bins, self.counts, self.errors,\
                    markersize=0, linestyle='None', color='darkgreen')
        ax.hist(self.raw_data, bins=self.Nbins, alpha=0.7, facecolor='green')
        
        x_for_fit_display = np.linspace(self.bins[0]*0.9, self.bins[-1]*1.1, 100)
        ax.plot(x_for_fit_display,\
                self.lognormal(self.p_result, x_for_fit_display),\
                color='black', lw=1, marker='None',\
                label= "$\mu_{log}\,=\,"+\
                           "{:.1f}".format(self.p_result["logmu"].value*mu_sf)+\
                           "\,"+mu_unit+"$\n"+
                       "$\sigma_{log}\,=\,"+\
                           "{:.1f}".format(self.p_result["logstd"].value*sigma_sf)+\
                           "\,"+sigma_unit+"$")

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel("$counts$")
#        ax.tick_params(axis='both', pad=6, width=1, length=2)
        ax.set_xlim([self.bins[0]-self.bin_width, self.bins[-1]+self.bin_width])
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(savename)
        print("Saved plot to " + savename)
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
        self.fitresults = lmfit.minimize(self.gaussian_residual, self.p,\
                        args=(self.bins, self.counts, self.errors))
        print(lmfit.fit_report(self.fitresults))
        self.p_result = self.fitresults.params
        
    def plot_gaussian(self, savename="GaussianFit.png",\
                        mu_unit="nm", mu_sf=1, sigma_unit="\%", sigma_sf=100):
        fig, ax = plt.subplots()
        ax.errorbar(self.bins, self.counts, self.errors,\
                    markersize=0, linestyle='None', color='darkgreen')
        ax.hist(self.raw_data, bins=self.Nbins, alpha=0.7, facecolor='green')
        
        x_for_fit_display = np.linspace(self.bins[0]*0.9, self.bins[-1]*1.1, 100)
        ax.plot(x_for_fit_display,\
                self.gaussian(self.p_result, x_for_fit_display),\
                color='black', lw=1, marker='None',\
                label= "$\mu\,=\,"+\
                           "{:.1f}".format(self.p_result["mu"].value*mu_sf)+\
                           "\,"+mu_unit+"$\n"+
                       "$\sigma\,=\,"+\
                           "{:.1f}".format(self.p_result["std"].value*sigma_sf)+\
                           "\,"+sigma_unit+"$")

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel("$counts$")
        ax.set_xlim([self.bins[0]-self.bin_width, self.bins[-1]+self.bin_width])
        ax.legend(loc='best')
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
        
    def load_xls(self, xlsfile, lengthidx=-1):
        xlsdata = open(xlsfile, "r")
        lengths = []
        for line in xlsdata:
            split_line = line.strip().split()
            try:
                first_entry_number = float(split_line[0])
            except ValueError:
                continue
            lengths.append(float(split_line[lengthidx]))
        xlsdata.close()
        return np.asarray(lengths)