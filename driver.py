import sys
import datetime

from ChemSys import chem_sys
from PySCF   import driver_scf as SCF
from PyMP2   import driver_mp2 as MP2
from PyCC    import driver_cc as CC
from Misc    import timing


@timing.time_fxn
def main():

    my_methods = ['-mp2','-ccd']


    print('-' * 120)
    print(' ',datetime.datetime.now())
    print('-' * 120)

    chem1 = chem_sys.Chem_sys('chem1.inp')


    print(SCF.driver_scf(chem1))
    print()

    if '-mp2' in sys.argv[1:]:
        print(MP2.driver_mp2(chem1))
        print()

    if '-ccd' in sys.argv[1:]:
        print(CC.driver_cc(chem1))
        print()

    for arg in sys.argv[1:]:
        if arg not in my_methods:
            print('ERROR ',arg,' is an invalid option')
            sys.exit()

    print('-' * 120)






if __name__ == '__main__':
    print(main())
