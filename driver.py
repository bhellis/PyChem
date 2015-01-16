from time import time
import datetime
from ChemSys import chem_sys
from PySCF import driver_scf as SCF
from PyMP2 import driver_mp2 as MP2
from PyCC import driver_cc as CC


def main():
    start = time() 
    print('-' * 140)
    print(datetime.datetime.now())
    print('-' * 140)

    chem1 = chem_sys.Chem_sys('chem1.inp')

    SCF.driver_scf(chem1)
    chem1.show_e_info('SCF')
    MP2.driver_mp2(chem1)
    chem1.show_e_info('MP2')
    CC.driver_cc(chem1)
    chem1.show_e_info('CC')

    print('-' * 140)
    print('Total time: %s' % (time() - start))


if __name__ == '__main__':
    main()
