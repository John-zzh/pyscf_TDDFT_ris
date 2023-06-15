#!/usr/bin/python
Hartree_to_eV = 27.211386245988

'''
GB Radii
Ghosh, Dulal C and coworkers
The wave mechanical evaluation of the absolute radii of atoms.
Journal of Molecular Structure: THEOCHEM 865, no. 1-3 (2008): 60-67.
'''

elements_106 = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

radii = [0.5292, 0.3113, 1.6283, 1.0855, 0.8141, 0.6513, 0.5428, 0.4652, 0.4071, 0.3618,
2.165, 1.6711, 1.3608, 1.1477, 0.9922, 0.8739, 0.7808, 0.7056, 3.293, 2.5419,
2.4149, 2.2998, 2.1953, 2.1, 2.0124, 1.9319, 1.8575, 1.7888, 1.725, 1.6654,
1.4489, 1.2823, 1.145, 1.0424, 0.9532, 0.8782, 3.8487, 2.9709, 2.8224, 2.688,
2.5658, 2.4543, 2.352, 2.2579, 2.1711, 2.0907, 2.016, 1.9465, 1.6934, 1.4986,
1.344, 1.2183, 1.1141, 1.0263, 4.2433, 3.2753, 2.6673, 2.2494, 1.9447, 1.7129,
1.5303, 1.383, 1.2615, 1.1596, 1.073, 0.9984, 0.9335, 0.8765, 0.8261, 0.7812,
0.7409, 0.7056, 0.6716, 0.6416, 0.6141, 0.589, 0.5657, 0.5443, 0.5244, 0.506,
1.867, 1.6523, 1.4818, 1.3431, 1.2283, 1.1315, 4.4479, 3.4332, 3.2615, 3.1061,
2.2756, 1.9767, 1.7473, 1.4496, 1.2915, 1.296, 1.1247, 1.0465, 0.9785, 0.9188,
0.8659, 0.8188, 0.8086]
exp = [1/(i*1.8897259885789)**2 for i in radii]

as_exp = dict(zip(elements_106,exp))
# print(as_exp)
RSH_F = [
'lc-b3lyp',
'wb97',
'wb97x',
'wb97x-d3',
'cam-b3lyp']


hybride_F = [
'b3lyp',
'tpssh',
'm05-2x',
'pbe0',
'm06',
'm06-2x',
None]
hybride_ax = [
0.2,
0.1,
0.56,
0.25,
0.27,
0.54,
1]
Func_ax = dict(zip(hybride_F, hybride_ax))



def gen_alpha_beta_ax(functional):

    '''
        RSH functionals have specific a_x, beta, alpha values;
        hybride fucntionals have fixed alpha12 and beta12 values,
        with different a_x values, by which create beta, alpha
    '''
    if functional in RSH_F:
        a_x = 1
    elif functional in hybride_F:
        a_x = Func_ax[functional]

    return a_x
