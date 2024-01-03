import numpy as np

import matplotlib.pyplot as plt
import argparse
import math
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.ticker as mtick
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "Ubuntu"
plt.rcParams["mathtext.fontset"] = "stix"
# from matplotlib.font_manager import fontManager

# fonts = set(f.name for f in fontManager.ttflist)
# print(fonts)

def str2bool(str):
    # print(str)
    if str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def gen_args():
    parser = argparse.ArgumentParser(description='plot sepctrum')
    parser.add_argument('-n', '--nstates', type=int, default=20, help='number of states')
    parser.add_argument('-FWHM', '--FWHM', type=float, default=0.6, help='full width at half maximum (FWHM), in eV')
    parser.add_argument('-g', '--grid', type=float, default=200, help='grind points')

    parser.add_argument('-lorentzian', '--lorentzian',   type=str2bool, default=False, help='lorentzian')
    parser.add_argument('-nm', '--eV2nm_broaden', type=str2bool, default=False, help='broaden in eV and then use nm unit')

    parser.add_argument('-f', '--files',   type=str, default=[], nargs='+', help='a lsit of spectra files path')
    parser.add_argument('-ftype', '--filetypes',   type=str, default=[], nargs='+', help='the corresponding lsit of file types')

    parser.add_argument('-name', '--molname',   type=str, default='', help='file name')
    parser.add_argument('-format', '--format',   type=str, default='pdf', help='png, pdf, eps')
    parser.add_argument('-dpi', '--dpi',   type=int, default=300, help='dpi, 300, 600')

    args = parser.parse_args()
    return args

args = gen_args()


def read_data(filename, filetype):
    if 'Gaussian' in filetype:
        # the gaussian output
        with open(filename, 'r') as file:
            lines = file.readlines()

        data = []
        for line in lines:
            if line.startswith(' Excited State'):
                parts = line.split()
                energy_ev = float(parts[4])
                # wavelength_nm = float(parts[6])
                oscillator_strength = float(parts[8][2:])
                data.append([energy_ev, oscillator_strength])
    
        data = np.array(data)

    elif 'lsqc' in filetype:
        # the gaussian output
        with open(filename, 'r') as file:
            lines = file.readlines()

        data = []
        for line in lines:
            if line.startswith('Local Excited State'):
                parts = line.split()
                energy_ev = float(parts[4])
                # wavelength_nm = float(parts[6])
                oscillator_strength = float(parts[8][2:])
                data.append([energy_ev, oscillator_strength])
    
        data = np.array(data)
        # print(data)

    elif 'ris' in filetype:
        # the pyscf-ris output
        data = np.loadtxt(filename, usecols=(0, 3), comments='#')

    elif 'sTD' in filetype:
        # the sTDA program output
        data = np.loadtxt(filename, usecols=(1, 2), skiprows=1, comments='#')

    elif 'Turbomole' in filetype:
        # the Turbomole output
        data = np.loadtxt(filename, usecols=(3, 7), comments='#')

    # print(data)
    return data

def get_line_style(filetype):
    LINEWIDTH = 1
    style_dict = {}
    style_dict['Gaussian-TDDFT'] = ('black','-')
    style_dict['Gaussian-TDA'] = ('black','-')

    style_dict['TDDFT-ris'] = ('red','--')
    style_dict['TDA-ris'] = ('red','--')

    style_dict['TDDFT-risp'] = ('#1f77b4','-')
    style_dict['TDA-risp'] = ('#1f77b4','-')

    style_dict['lsqc-TDDFT'] = ('#e377c2','-.')
    style_dict['lsqc-TDA'] = ('#e377c2','-.')

    style_dict['sTDDFT'] = ('#e377c2','-')
    style_dict['sTDA'] = ('#e377c2','-')
    
    style_dict['Turbomole-TDDFT'] = ('#2ca02c','-')
    style_dict['Turbomole-TDA'] = ('#2ca02c','-')

    line_color, linestyle = style_dict[filetype]

    return line_color, linestyle, LINEWIDTH


class input_file(object):
    def __init__(self, filename, filetype):
        self.filename = filename
        self.filetype = filetype
        self.data = read_data(filename, filetype)
        self.style = get_line_style(filetype)

def get_gaussian_broad(energy, os_stren, FWHM = args.FWHM):
    '''
    map (energy,os_stren) (1d_array,1d_array) pairs to gaussain lines

    g(x) = a * exp(-(x-b)**2/(2c**2))
    let int g(x) = ac * sqrt(2π) = os_stren
    full width at half maximum (FWHM) = 2sqrt(2ln(2)) * c
    '''
    start = max(min(energy)-1, 0)
    # start = 0
    end = max(energy)+1
    spacing = (end-start)/args.grid
    x = np.arange(start, end, spacing)
    y = np.zeros_like(x)

    c = FWHM/(2*(2*np.log(2)))

    for i in range(energy.shape[0]):
        a = os_stren[i]/(math.sqrt(2*math.pi)*c)
        current_peak = a * np.exp(-(x - float(energy[i]))**2/(2*c**2))
        # print('use integral to check the error')
        # print(spacing*np.sum(current_peak) - os_stren[i])
        y += current_peak
    return x, y

def get_lorentzian_broad(energy, os_stren, HWHM = 0.5*args.FWHM):

    '''
    l(x) = A /π * {HWHM/[(x-x0^2)+(HWHM)^2]}
    let int l(x) = A = os_stren
    half width at half maximum (HWHM)
    '''
    # start = max(min(energy)-1, 0)
    start = 0
    end = max(energy)+1
    spacing = (end-start)/args.grid
    x = np.arange(start, end, spacing)
    y = np.zeros_like(x)

    for i in range(energy.shape[0]):
        A = os_stren[i]
        current_peak = A/math.pi * (HWHM/(np.power(x-energy[i],2) + HWHM**2))
        # use integral to check the error
        # print("{:.3f}".format(spacing*np.sum(current_peak) - os_stren[i]))
        y += current_peak

    return x, y



def overlap_plot():
    # creat a figure instance
    fig, ax = plt.subplots(figsize=(4, 2))
    fig.subplots_adjust(left=0.12, right=0.7, bottom=0.16, top=0.98)
    
    for i in range(len(args.files)):
        print(f"plotting file No. {i}")
        filename, filetype = args.files[i], args.filetypes[i]
        print('file_name =', filename)
        print('file_type =', filetype)

        file_obj = input_file(filename, filetype)

        eV, os_stren = file_obj.data[:,0], file_obj.data[:,1]
        # print(os_stren)
        line_color, linestyle, linewidth = file_obj.style

        if args.lorentzian:
            print('lorentzian broadening')
            x, y = get_lorentzian_broad(eV, os_stren)
        else:
            print('gaussian broadening')
            x, y = get_gaussian_broad(eV, os_stren)

        if args.eV2nm_broaden:
            x = 1240/x
            x = np.flip(x)
            y = np.flip(y)

        ax.plot(x, y, label=filetype, color = line_color, linestyle=linestyle, linewidth=linewidth)

    if args.eV2nm_broaden:
        xlabel = 'Wavelength [nm]'
    else:
        xlabel = 'Energy [eV]'

    ax.set_xlabel(xlabel,y=0.20, fontsize=8)
    ax.set_ylabel(r"$\sigma\ [\mathrm{bohr}^2]$", fontsize=8)
    ax.tick_params(axis='both', which='both', labelsize=8, direction='in', pad=2)

    ax.legend(frameon=False, 
              fontsize=8, 
              handlelength=1.5,
              loc = 'upper left',
              bbox_to_anchor=(0.98,1.0),
              labelspacing=0.3) 

    # if args.format == 'png':
    #     plt.savefig(args.molname+'_UV.png', dpi=args.dpi)
    # elif args.format == 'pdf':
    #     plt.savefig(args.molname+'_UV.pdf')
    # elif args.format == 'eps':
    #     plt.savefig(args.molname+'_UV.eps')

    plt.savefig(args.molname+'_UV.' + args.format, dpi=args.dpi)

if __name__ == "__main__":
    overlap_plot()
