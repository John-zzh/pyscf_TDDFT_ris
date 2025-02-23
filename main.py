from pyscf_TDDFT_ris import readMO
import argparse, os
import time


def gen_args():

    def str2bool(str):
        if str.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif str.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser(description='Davidson')
    parser.add_argument('-f',    '--filename',    type=str,   default=None,      help='.fch filename (molecule.fch)')
    parser.add_argument('-fout', '--spectraoutname', type=str,   default=None,      help='output file name')
    parser.add_argument('-func', '--functional',  type=str,   default=None,      help='functional name (pbe0)')
    parser.add_argument('-b',    '--basis',       type=str,   default=None,      help='basis set name (def2-SVP)')
    parser.add_argument('-ax',   '--a_x',         type=float, default=None,      help='HF component in the hybrid functional')
    parser.add_argument('-w',    '--omega',       type=float, default=None,      help='screening factor in the range-separated functional')
    parser.add_argument('-alpha',   '--alpha',       type=float, default=None,      help='alpha in the range-separated functional')
    parser.add_argument('-beta',   '--beta',        type=float, default=None,      help='beta in the range-separated functional')
    
    parser.add_argument('-theta',   '--theta',       type=int,   default=0.2,       help='exponent = theta/R^2, optimal theta = 0.2')
    parser.add_argument('-J_fit',   '--J_fit',       type=str,  default='sp',   choices=['s', 'sp', 'spd'],  help='J fitting basis')
    parser.add_argument('-K_fit',   '--K_fit',       type=str,  default='s',    choices=['s', 'sp', 'spd'],  help='K fitting basis')
    parser.add_argument('-M',      '--max_mem_mb',   type=int,  default=8000,     help='maximum memory in MB')

    parser.add_argument('--group_size',              type=int,  default=10000,     help='group_size in AO direction')
    parser.add_argument('--group_size_aux',          type=int,  default=256,     help='group_size in auxAO direction')

    parser.add_argument('-Ktrunc', '--Ktrunc',       type=float,  default=40,     help='eV truncaion threshold for the MO in K')

    parser.add_argument('-GPU', '--GPU',           type=str2bool,  default=False,     help='use GPU')

    parser.add_argument('-TDA',  '--TDA',               type=str2bool,  default=False,    help='peform TDA calculation instead of TDDFT') 
    parser.add_argument('-n',    '--nroots',            type=int,   default=10,        help='the number of states you want to solve')
    parser.add_argument('-t',    '--conv_tol',          type=float,  default=1e-3,      help='the convengence tolerance in the Davidson diagonalization')
    parser.add_argument('-i',    '--max_iter',          type=int,   default=20,        help='the number of iterations in the Davidson diagonalization')
    parser.add_argument('-pt',   '--print_threshold',   type=float,   default=0.05,        help='the threshold of printing the transition coefficients')
    parser.add_argument('-GS',    '--GS',              type=str2bool,   default=False,    help='use Gram-Schdmit orthogonalization in Davidson diagonalization')
    parser.add_argument('-spectra', '--spectra',        type=str2bool,   default=True,    help='print out the spectra file')
    parser.add_argument('-single', '--single',          type=str2bool,   default='True', help='use single precision')

    # parser.add_argument('-CSF', '--CSF_trunc',          type=str2bool,   default=False,    help='truncate the CSF basis to speedup the calculation')
    
    # parser.add_argument('-specw', '--spectra_window',   type=float,   default=10.0,    help='the window of the spectra up to, in eV')
    # parser.add_argument('-pt2_tol', '--pt2_tol',        type=float,   default=1e-4, help='the threshold of S-CSF PT2 evaluation')
    # parser.add_argument('-N', '--N_cpus',               type=int,   default=10, help='the number of CPUs to use')
    
    # parser.add_argument('-approx', '--approximation',   type=str,   default='ris', help='ris sTDA')

    args = parser.parse_args()

    if args.filename == None:
        raise ValueError('I need the .fch filename, such as -f molecule.fch')
    if args.functional == None and args.a_x == None and args.omega == None:
        raise ValueError('I need the functional name, such as -func pbe0; or functional parameters: a_x, omega, alpha, beta, such as -ax 0.25; -w 0.5 -al 0.5 -be 1.0')
    if args.spectraoutname == None:
        basename_with_ext = os.path.basename(args.filename)
        args.spectraoutname = os.path.splitext(basename_with_ext)[0]

    return args

args = gen_args()

print('args.nroots', args.nroots)

if __name__ == '__main__':
    start = time.time()
    print('Woring directory:', os.getcwd())
    '''
    if mf object already has a functional name, 
    then do not need to specify the a_x or (omega, alpha, beta), 
    as they will be read from the build-in library

    if manually specify the a_x or (omega, alpha, beta), then you dont have to input the functional name 
    '''

    ''' if args.filename ends with string .fch, use  get_mf_from_fch function '''

    if '.fch' in args.filename:
        mf = readMO.get_mf_from_fch(fch_file=args.filename, 
                                    functional=args.functional)
        get_mf_from_fch_time = time.time() - start
        print(f'get_mf_from_fch_time: {get_mf_from_fch_time:.2f} seconds')
    elif 'molden' in args.filename:
        print('!!!!!!!!!!!warning: molden is not the best format to read MO. Pleas use MOKIT to convert your file to fch format!!!!!!!!!!!!!')
        if args.basis == None:
            raise ValueError('I need the basis set name, such as -b def2-TZVP. Because molden file does not provide correct basis set information.')
        mf = readMO.get_mf_from_molden(molden_file=args.filename, 
                                       functional=args.functional,
                                       basis=args.basis)

    if args.GPU == False:
        from pyscf_TDDFT_ris import TDDFT_ris_Ktrunc as TDDFT_ris
        # print('use new code')
        td = TDDFT_ris.TDDFT_ris(mf=mf, 
                        theta=args.theta,
                        J_fit=args.J_fit, 
                        K_fit=args.K_fit,
                        a_x=args.a_x,
                        omega=args.omega,
                        alpha=args.alpha,
                        beta=args.beta,
                        Ktrunc=args.Ktrunc,
                        max_mem_mb=args.max_mem_mb,
                        conv_tol=args.conv_tol,
                        nroots=args.nroots, 
                        single=args.single,
                        GS=args.GS,
                        max_iter=args.max_iter,
                        out_name=args.spectraoutname,
                        spectra = args.spectra,
                        print_threshold=args.print_threshold)           
    else:
        from pyscf_TDDFT_ris_cupy import TDDFT_ris
        print('use GPU code')
        td = TDDFT_ris.TDDFT_ris(mf=mf.to_gpu(), 
                        theta=args.theta,
                        J_fit=args.J_fit, 
                        K_fit=args.K_fit,
                        a_x=args.a_x,
                        omega=args.omega,
                        alpha=args.alpha,
                        beta=args.beta,
                        Ktrunc=args.Ktrunc,
                        conv_tol=args.conv_tol,
                        nroots=args.nroots, 
                        single=args.single,
                        group_size=args.group_size,
                        group_size_aux=args.group_size_aux,
                        GS=args.GS,
                        max_iter=args.max_iter,
                        out_name=args.spectraoutname,
                        spectra = args.spectra,
                        print_threshold=args.print_threshold)         

    if args.TDA == True:
        energies, X, oscillator_strength = td.kernel_TDA()
    else:
        energies, X, Y, oscillator_strength = td.kernel_TDDFT()
    end = time.time()
    print(f'total ris time: {end-start:.2f} seconds')

    citation_info = """
    Please cite the TDDFT-ris method:

        1.  Zhou, Zehao, Fabio Della Sala, and Shane M. Parker. 
            Minimal auxiliary basis set approach for the electronic excitation spectra 
            of organic molecules. The Journal of Physical Chemistry Letters 
            14, no. 7 (2023): 1968-1976.
            (must cite)

        2.  Zhou, Zehao, and Shane M. Parker. 
            Converging Time-Dependent Density Functional Theory Calculations in Five Iterations 
            with Minimal Auxiliary Preconditioning. Journal of Chemical Theory and Computation 
            20, no. 15 (2024): 6738-6746. 
            (for efficient orbital truncation technique)

        2.  Giannone, Giulia, and Fabio Della Sala. 
            Minimal auxiliary basis set for time-dependent density functional theory and 
            comparison with tight-binding approximations: Application to silver nanoparticles.
            The Journal of Chemical Physics 153, no. 8 (2020).
            (The idea of TDDFT-ris originates from TDDFT-as)

    And cite the pyscf-TDDFT-ris package:

        1. Zehao Zhou, pyscf-TDDFT-ris, (https://github.com/John-zzh/pyscf_TDDFT_ris)

        2. Jingxiang Zou, Molecular Orbital Kit (MOKIT), (https://gitlab.com/jxzou/mokit)

        3. PySCF: the Python-based simulations of chemistry framework, 
           Q. Sun, et. al., and G. K.-L. Chan, WIREs Comput. Mol. Sci. 8, e1340 (2018) 
           (https://pyscf.org/about.html)
    """

    print(citation_info)