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
    parser.add_argument('--nstates', type=int, default=20, help='number of states')
    parser.add_argument('--FWHM', type=float, default=0.6, help='full width at half maximum (FWHM), in eV')
    parser.add_argument('--grid', type=float, default=200, help='grind points')

    parser.add_argument('--eV2nm_broaden', type=str2bool, default=False, help='broaden in eV and then use nm unit')

    parser.add_argument('--xlimit',    type=float,default=[], nargs='+',  help='limit of x axis, in eV or nm')
    parser.add_argument('--eV_file',   type=str,  default=[], nargs='+', help='eV file name')
    parser.add_argument('--CD_file',   type=str,  default=[], nargs='+', help='CD file name')
    parser.add_argument('--g_file',    type=str,  default=[], nargs='+',  help='g file name')
    parser.add_argument('--ylabel',    type=str,   help='')

    parser.add_argument('--filetypes',   type=str, default=[], nargs='+', help='the corresponding lsit of file types')


    parser.add_argument('--outname',   type=str, default='', help='output file name')
    parser.add_argument('--format',   type=str, default='pdf', help='png, pdf, eps')
    parser.add_argument('--dpi',   type=int, default=300, help='dpi, 300, 600')

    args = parser.parse_args()
    return args

args = gen_args()


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

    style_dict['lsqc-TDDFT-risp'] = ('#e377c2','-.')
    style_dict['lsqc-TDA-risp'] = ('#e377c2','-.')

    style_dict['GEBF-TDDFT'] = ('#ed0771','-')
    style_dict['GEBF-TDA'] = ('#e377c2','-')

    style_dict['GEBF-TDDFT-risp'] = ('#071fed','--')
    style_dict['GEBF-TDA-risp'] = ('#071fed','--')


    style_dict['CSF-TDDFT-ris'] = ('#2ca02c','-')
    style_dict['CSF-TDA-ris'] = ('#2ca02c','-')

    style_dict['sTDDFT'] = ('#e377c2','-')
    style_dict['sTDA'] = ('#e377c2','-')
    
    style_dict['Turbomole-TDDFT'] = ('#2ca02c','-')
    style_dict['Turbomole-TDA'] = ('#2ca02c','-')

    line_color, linestyle = style_dict[filetype]

    return line_color, linestyle, LINEWIDTH



def get_gaussian_broad(energy, y_value, FWHM = args.FWHM):
    '''
    map (energy,y_value) (1d_array,1d_array) pairs to gaussain lines

    g(x) = a * exp(-(x-b)**2/(2c**2))
    let int g(x) = ac * sqrt(2π) = y_value
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
        a = y_value[i]/(math.sqrt(2*math.pi)*c)
        current_peak = a * np.exp(-(x - float(energy[i]))**2/(2*c**2))

        y += current_peak
    return x, y

def overlap_plot():
    # creat a figure instance
    fig, ax = plt.subplots(figsize=(4, 2))
    fig.subplots_adjust(left=0.12, right=0.7, bottom=0.16, top=0.98)
    
    for i in range(len(args.eV_file)):
        print(f"plotting file No. {i}")

        eV = np.loadtxt(args.eV_file[i])
        if args.CD_file:
            y_value = np.loadtxt(args.CD_file[i])
        elif args.g_file:
            y_value = np.loadtxt(args.g_file[i])
        # print(eV[:args.nstates], y_value[:args.nstates])
        x, y = get_gaussian_broad(eV[:args.nstates], y_value[:args.nstates])

        if args.eV2nm_broaden:
            x = 1240/x
            x = np.flip(x)
            y = np.flip(y)

        filetype = args.filetypes[i]
        line_color, linestyle, linewidth = get_line_style(filetype)
        ax.plot(x, y, label=filetype, color = line_color, linestyle=linestyle, linewidth=linewidth)

    if args.eV2nm_broaden:
        xlabel = 'Wavelength [nm]'
    else:
        xlabel = 'Energy [eV]'

    if args.xlimit:
        ax.set_xlim(args.xlimit[0], args.xlimit[1])

    ax.set_xlabel(xlabel,y=0.20, fontsize=8)
    ax.set_ylabel(args.ylabel, fontsize=8)
    ax.tick_params(axis='both', which='both', labelsize=8, direction='in', pad=2)

    ax.legend(frameon=False, 
              fontsize=8, 
              handlelength=1.5,
              loc = 'upper left',
              bbox_to_anchor=(0.98,1.0),
              labelspacing=0.3) 

    plt.savefig(f'{args.outname}.{args.format}', dpi=args.dpi)

if __name__ == "__main__":
    overlap_plot()
