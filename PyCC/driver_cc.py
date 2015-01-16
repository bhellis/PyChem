import numpy as NP
from ChemSys import chem_sys
from PyMP2 import driver_mp2 as MP2


def driver_cc(my_sys):
    nocc  = my_sys.get_nocc()
    nspin = my_sys.get_nspin()
    hcore = my_sys.get_hcore()
    vee   = my_sys.get_vee()
    orb_eng = my_sys.get_orb_eng()

    t2 = MP2.calc_t_mp2(nocc, nspin, hcore, vee, orb_eng)
    t1 = NP.zeros((nocc, nspin), dtype=NP.float64)
    my_sys.set_e_corr(calc_cc_ecorr(nocc, nspin, vee, t1, t2))


def calc_cc_ecorr(nocc, nspin, vee, t1, t2):
    #leaving out FOCK for now
    ecorr = 0.0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nocc, nspin):
                for b in range(nocc, nspin):
                    t_value = t2[i,j,a,b] + (t1[i,a] * t1[j,b]) - (t1[i,b] * t1[j,a])
                    ecorr += (vee[i,a,j,b] - vee[i,b,j,a]) * t_value
    return 0.25 * ecorr
                    


