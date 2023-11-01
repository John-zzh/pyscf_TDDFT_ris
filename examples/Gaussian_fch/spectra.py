import numpy as np

import matplotlib.pyplot as plt
import argparse
import math
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.ticker as mtick
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

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
    parser.add_argument('-FWHM', '--FWHM', type=float, default=0.6, help='full width at half maximum (FWHM), in cm-1 or eV')
    # parser.add_argument('-HWHM', '--HWHM', type=float, default=0.1, help='half width at half maximum (FWHM), in cm-1 or eV')
    parser.add_argument('-g', '--grid', type=float, default=200, help='grind points')
    parser.add_argument('-eV2nm', '--eV2nm_broaden', type=str2bool, default=False, help='broaden in eV and then use nm unit')
    parser.add_argument('-f', '--files',   type=str, default=None, nargs='+', help='a lsit of data files')

    parser.add_argument('-fsize', '--fsize',   type=float, default=[4,2.2], nargs='+', help='figure size')
    parser.add_argument('-left', '--left',      type=float, default=None, help='left margin')
    parser.add_argument('-right', '--right',   type=float, default=None,  help='right margin')
    parser.add_argument('-top', '--top',        type=float, default=None,  help='top margin')
    parser.add_argument('-bottom', '--bottom',  type=float, default=None,  help='bottom margin')

    parser.add_argument('-x', '--xlim',   type=float, default=-1, help='200nm, 250nm')
    parser.add_argument('-xe', '--xlimend',   type=float, default=-1, help='500nm, 550nm')

    parser.add_argument('-xns', '--xnormstart',   type=float, default=0, help='200nm, 250nm')
    parser.add_argument('-xne', '--xnormend',   type=float, default=6, help='200nm, 250nm')

    parser.add_argument('-ylim', '--ylim',   type=float, default=0, help='enlarge y axis')

    parser.add_argument('-e', '--experimental',   type=str, default=None, help='experimental.txt')
    parser.add_argument('-cd', '--cd_spectra',   type=str2bool, default=False, help='cd spectra')
    parser.add_argument('-m', '--mol',   type=str, default='', help='molecule name')
    parser.add_argument('-p', '--molpic',   type=str, default='', help='picture filename')
    parser.add_argument('-px', '--molpicx',   type=float, default=1, help='picture x position')
    parser.add_argument('-py', '--molpicy',   type=float, default=1, help='picture y position')
    parser.add_argument('-z', '--zoom',   type=float, default=1, help='picture zoom scale')
    parser.add_argument('-t', '--title',   type=str2bool, default=True, help='figure title')
    parser.add_argument('-l', '--legend',   type=str2bool, default=True, help='legend')
    parser.add_argument('-legendsize', '--legendsize',   type=int, default=7, help='legendsize')

    parser.add_argument('-log', '--log',   type=str2bool, default=False, help='log scale')
    parser.add_argument('-linewidth', '--linewidth',   type=float, default=1.25, help='log scale')
    parser.add_argument('-xlabel', '--xlabel',   type=str2bool, default=True, help='x label on or off')
    parser.add_argument('-ylabel', '--ylabel',   type=str2bool, default=True, help='y label on or off')
    parser.add_argument('-lorentzian', '--lorentzian',   type=str2bool, default=False, help='lorentzian')
    parser.add_argument('-format', '--format',   type=str, default='png', help='png, pdf')
    parser.add_argument('-dpi', '--dpi',   type=int, default=300, help='dpi, 300, 600')


    # parser.add_argument('-gaussian', '--gaussian',   type=str2bool, default=True, help='gaussian')

    args = parser.parse_args()
    return args

args = gen_args()

# FWHM = args.FWHM/219474.6

def gen_gaussian(energy, os_stren, FWHM = args.FWHM):
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
        # use integral to check the error
        # print(spacing*np.sum(current_peak) - os_stren[i])
        y += current_peak
    return x, y

def gen_lorentzian(energy, os_stren, HWHM = 0.5*args.FWHM):

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



def read_eV_os(tmp):
    data = np.loadtxt(tmp, skiprows=1, usecols = (0,3))

    max_rows = args.nstates
    if max_rows != 0:
        # only plot first nstates
        eV = data[:max_rows,0]
        os_stren = data[:max_rows,1]

    else:
        # use all available states
        eV = data[:,0]
        os_stren = data[:,1]
    return eV, os_stren

def gen_style(tmp):

    method_color = '#2ca02c'
    linewidth = 1.0
    linestyle='-.'

    gloabl_lw = args.linewidth

    if tmp == 'TDDFT':
        method_color = 'black'
        linewidth = gloabl_lw
        linestyle='-'
    if tmp == 'sTDDFT':
        method_color = '#1f77b4' #blue
        linewidth = gloabl_lw
        linestyle='--'
    if tmp == 'TDDFT-s' or tmp == 'cd_TDDFT-s' or tmp == 'cd_os' or tmp == 'TDDFT-s_1.00':
        method_color = 'red'
        linewidth = gloabl_lw
        # linestyle='-.'
        linestyle='-'
    if tmp == 'TDDFT-sp' or tmp == 'TDDFT-s_0.05':
        # method_color = '#BBBDB7' #gray
        method_color = '#007F0E' #green
        linewidth = gloabl_lw
        linestyle='-.'
        # linestyle='dotted'
    if tmp == 'TDDFT-spd' or tmp == 'TDDFT-s_0.25':
        method_color = '#e377c2'
        linewidth = gloabl_lw
        linestyle='-'
    if tmp == 'TDDFT-s_vacuum':
        method_color = '#2ca02c'
        linewidth = gloabl_lw
        linestyle='-'
    # if tmp == 'TDDFT-s_vaccum':
    #     method_color = '#e68f66'
    #     linewidth = 1.0
    #     linestyle='-'


    return method_color, linewidth, linestyle

def getImage(path, zoom=args.zoom):
    return OffsetImage(plt.imread(path), zoom=zoom)

legen_dict={}
legen_dict['TDDFT'] = 'TDDFT'
legen_dict['sTDDFT'] = 'sTDDFT'
legen_dict['TDDFT-ris_UV_spectra.txt'] = 'TDDFT-ris'
legen_dict['TDDFT-sp'] = 'TDDFT-risp'

def overlap_plot():
    fig, ax = plt.subplots(figsize=(args.fsize[0], args.fsize[1])) # creat a figure instance

    if args.xlabel and args.ylabel:
    # catene-1b
        left=0.16
        right=0.98
        bottom=0.19
        top=0.95
    elif args.ylabel and not args.xlabel:
    # retinal  BF2WS3 Betaine 30
        left=0.16
        right=0.98
        bottom=0.10
        top=0.95
    elif not args.ylabel and not args.xlabel:
    # all molecules
        left=0.10
        right=0.98
        bottom=0.10
        top=0.98

    if args.left:
        left = args.left
    if args.right:
        right = args.right
    if args.bottom:
        bottom = args.bottom
    if args.top:
        top = args.top
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


    if args.experimental:
        data = np.loadtxt(args.experimental)
        if args.eV2nm_broaden:
            x = data[:,0]
            y = data[:,1]
        else:
            '''nm to eV'''
            x = 1240/data[:,0]
            y = data[:,1]
            x = np.flip(x)
            y = np.flip(y)

        experimental_y_max = np.max(data[:,1])
        print('experimental_y_max =', experimental_y_max)
        ax.plot(x, y, label='experimental', color = 'black', linewidth=1.8, linestyle='-')

    y_max = 0
    y_min = 0
    if args.lorentzian:
        gen_broadening_func = gen_lorentzian
    else:
        gen_broadening_func = gen_gaussian


    # max_energy = 1000
    for file in args.files:
        print('file_name =', file)
        eV, os_stren = read_eV_os(file)
        if file == 'TDDFT':
            max_energy = eV[-1]

        x, y = gen_broadening_func(eV, os_stren)
        # if args.log:
        #     y = np.log10(y)
        if args.eV2nm_broaden:
            x = 1240/x
            x = np.flip(x)
            y = np.flip(y)

        xnormstart = np.sum(x <= args.xnormstart)
        xnormend = np.sum(x <= args.xnormend)

        print('xnormstart', x[xnormstart-1])
        print('xnormend', x[xnormend-1])
        print('y_max =', np.max(y))
        print('y_min =', np.min(y))


        if args.experimental:
            # y = (experimental_y_max/np.max(y[xnormstart:xnormend]))*y
            y = y/np.max(y[xnormstart:xnormend]) * experimental_y_max

            print('max_os =', np.max(y[xnormstart:xnormend]))
            y_max = max(abs(y_max), abs(np.max(y[xnormstart:xnormend])))

        y_max = max(y_max, np.max(y))
        y_min = min(y_min, np.min(y))


        method_color, linewidth, linestyle = gen_style(file)

        ax.plot(x, y, label=legen_dict[file], color = method_color, linewidth=linewidth, linestyle=linestyle)


        # ax.set_yscale('linear')
    # print('y_min =', y_min)
    if args.molpic:
        ab = AnnotationBbox(getImage(args.molpic), (args.molpicx, args.molpicy), frameon=False)
        ax.add_artist(ab)



    if args.xlim >= 0:
        if args.xlimend < 0:
            print('max_energy = ', max_energy)
            plt.xlim(args.xlim, max_energy)
        elif args.xlimend > 0:
            plt.xlim(args.xlim, args.xlimend)
        elif args.xlimend == 0:
            plt.xlim(args.xlim, )



    y_lim_end   = y_max*1.3 if args.ylim == 0 else args.ylim
    y_lim_start = y_min*1.35 if args.cd_spectra else 0

    if args.log:
        ax.set_yscale('log',base=10)
        y_lim_start = 0.01
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    else:
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    print('y_lim_start =', y_lim_start)
    print('y_lim_end =', y_lim_end)

    plt.ylim(y_lim_start, y_lim_end)


    plt.minorticks_on()
    # ax.ticklabel_format(style='plain')
    if args.xlabel:
        if args.eV2nm_broaden:
            xlabel = 'Wavelength [nm]'
        else:
            xlabel = 'Energy [eV]'
        ax.set_xlabel(xlabel,y=0.20, fontsize=10)

    if args.ylabel:
        if args.cd_spectra:
            ylabel = 'Rotatory strength'
        elif args.experimental:
            ylabel = 'Normalized absorbance [a.u.]'
        else:
            ylabel = r"$\sigma\ [\mathrm{bohr}^2]$"

        ax.set_ylabel(ylabel, fontsize=10)
    if args.legend:
        ax.legend(frameon=False, loc = 'upper right', fontsize=args.legendsize, labelspacing=0.1) #bbox_to_anchor=(1.02,1.05),
    if args.format == 'png':
        plt.savefig(args.mol+'_UV.png' if not args.cd_spectra else args.mol+'_CD.png', dpi=args.dpi)
    elif args.format == 'pdf':
        plt.savefig(args.mol+'_UV.pdf' if not args.cd_spectra else args.mol+'_CD.pdf')
    elif args.format == 'eps':
        plt.savefig(args.mol+'_UV.eps')

if __name__ == "__main__":
    overlap_plot()
