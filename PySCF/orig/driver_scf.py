import numpy as np
from PySCF import scf_routines as SCF
from PySCF import integral_reader as IntRd
from PySCF import chem_sys
from PySCF import ao_2_mo
from ChemSys import my_sys


def driver_scf():
    mysys = chem_sys.Chem_sys('chem1.inp')
    hcore = mysys.get_scale_hcore() * IntRd.read_hcore('hcore1.fmt')
    vee   = mysys.get_scale_vee()   * IntRd.read_vee('vee1.fmt')
    smat  = IntRd.read_smat('smat1.fmt')
    vnuc  = SCF.calc_vnuc('geom1.fmt')

    # get guess density / energy
    pmat = SCF.calc_pmat(mysys, hcore, smat)
    fmat = SCF.calc_fock(mysys, hcore, vee, pmat)
    e_hf = SCF.calc_energy(mysys, hcore, fmat, pmat)
    norm_last = np.linalg.norm(pmat)
    print('SCF: Guess Energy: %s' % (e_hf + vnuc))

    # SCF cycle
    ncycle = 0
    while True:
        ncycle += 1
        pmat = SCF.calc_pmat(mysys, fmat, smat)
        norm_new = np.linalg.norm(pmat)
        fmat = SCF.calc_fock(mysys, hcore, vee, pmat)
        e_hf = SCF.calc_energy(mysys, hcore, fmat, pmat)
        if abs(norm_last - norm_new) < mysys.pmat_cutoff or ncycle == 1000:
            break
        else:
            norm_last = norm_new
    print('SCF: Final Energy: %s -- %s cycles' % ((e_hf + vnuc), ncycle))

    # AO to MO Transformation
    hcoremo, veemo = ao_2_mo.transform_ao2mo(mysys, fmat, smat, hcore, vee)
    e_hf = ao_2_mo.calc_ehf(mysys, hcoremo, veemo) + vnuc
    print('MO: Final Energy: %s' % e_hf)
