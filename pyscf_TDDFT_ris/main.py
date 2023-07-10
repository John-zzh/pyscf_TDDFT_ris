from pyscf_TDDFT_ris import TDDFT_ris, readMO, parameter
import argparse

def str2bool(str):
    if str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def gen_args():
    parser = argparse.ArgumentParser(description='Davidson')
    parser.add_argument('-f',    '--fchfile',     type=str,   default='*.fch',   help='.fch filename (molecule.fch)')
    parser.add_argument('-func', '--functional',  type=str,   default=None,      help='functional filename (pbe0)')
    parser.add_argument('-ax',   '--a_x',         type=float, default=None,      help='HF component in the hybrid functional')
    parser.add_argument('-w',    '--omega',       type=float, default=None,      help='screening factor in the range-separated functional')
    parser.add_argument('-al',   '--alpha',       type=float, default=None,      help='alpha in the range-separated functional')
    parser.add_argument('-be',   '--beta',        type=float, default=None,      help='beta in the range-separated functional')
    
    parser.add_argument('-th',   '--theta',       type=int,   default=0.2,       help='exponent = theta/R^2, optimal theta = 0.2')
    parser.add_argument('-p',    '--add_p',       type=bool,  default=False,     help='add an extra p function to the auxilibary basis')

    parser.add_argument('-tda',  '--TDA',         type=bool,   default=False,     help='peform TDA calculation instead of TDDFT') 
    parser.add_argument('-n',    '--nroots',      type=int,   default=20,        help='the number of states you want to solve')
    parser.add_argument('-t',    '--conv_tol',    type=bool,  default=1e-5,      help='the convengence tolerance in the Davidson diagonalization')
    parser.add_argument('-i',    '--max_iter',    type=int,   default=20,        help='the number of iterations in the Davidson diagonalization')

    args = parser.parse_args()


    if args.functional == None:
        if args.a_x == None and args.omega == None and args.alpha == None and args.beta == None:
            raise ValueError('Please specify the functional name or the functional parameters')
        else:
            if a_x:
                args.a_x = a_x
                print("hybrid functional")
                print(f"manually input HF component ax = {a_x}")

            elif omega and alpha and beta:
                args.a_x = 1
                args.omega = omega
                args.alpha = alpha
                args.beta = beta
                print("range-separated hybrid functional")
                print(f"manually input ω = {args.omega}, screening factor")
                print(f"manually input α = {args.alpha}, fixed HF exchange contribution")
                print(f"manually input β = {args.beta}, variable part")

            else:
                raise ValueError('missing parameters for range-separated functional, please input (w, al, be)')

    elif args.functional != None:
        functional = args.functional.lower()
        args.functional = functional
        print('Loading defult functional paramters from parameter.py.')

        if functional in parameter.rsh_func.keys():
            '''
            RSH functional, need omega, alpha, beta
            '''
            print('use range-separated hybrid functional')
            omega, alpha, beta = parameter.rsh_func[functional]
            args.a_x = 1
            args.omega = omega
            args.alpha = alpha
            args.beta = beta

        elif functional in parameter.hbd_func.keys():
            print('use hybrid functional')
            args.a_x = parameter.hbd_func[functional]

        else:
            raise ValueError(f"I do not have paramters for functional {mf.xc} yet, please either manually input HF component a_x or add parameters in the parameter.py file.")

    return args

args = gen_args()


if __name__ == '__main__':
    

    '''
    if mf object already has a functional name, 
    then do not need to specify the a_x or (omega, alpha, beta), 
    as they will be read from the build-in library

    if manually specify the a_x or (omega, alpha, beta), then you dont have to input the functional name 
    '''
    mf = readMO.get_mf_from_fch(fch_file = args.fchfile, 
                                functional = args.functional)
    
    td = TDDFT_ris.TDDFT_ris(mf=mf, 
                            theta=args.theta,
                            add_p=args.add_p, 
                            a_x=args.a_x,
                            omega=args.omega,
                            alpha=args.alpha,
                            beta=args.beta,
                            conv_tol=args.conv_tol,
                            nroots=args.nroots, 
                            max_iter=args.max_iter)


    if args.TDA == True:
        energies, X, oscillator_strength = td.kernel_TDA()
    else:
        energies, X, Y, oscillator_strength = td.kernel_TDDFT()
