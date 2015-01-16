

class Chem_sys(object):
    def __init__(self, filename):
        self.nocc, self.nspace, self.pmat_cutoff, self.scale_hcore, self.scale_vee  = \
        _read_input(filename)

        if self.nocc % 2 != 0:
            print('Error --> This is an RHF code, Nocc must be even --> Exiting...')
            sys.exit()

        self.nspin = 2 * self.nspace

    def get_nocc(self):
        return self.nocc
    
    def get_nspace(self):
        return self.nspace
    
    def get_pmat_cutoff(self):
        return self.pmat_cutoff
    
    def get_scale_hcore(self):
        return self.scale_hcore
    
    def get_scale_vee(self):
        return self.scale_vee


def _read_input(filename):
    with open(filename) as f:
        lines = f.readlines()
    thisline = lines[0].split()
    nocc         = int(thisline[1])
    thisline = lines[1].split()
    nspace       = int(thisline[1])
    thisline = lines[2].split()
    pmat_cutoff  = float(thisline[1])
    thisline = lines[3].split()
    scale_hcore  = float(thisline[1])
    thisline = lines[4].split()
    scale_vee    = float(thisline[1])
    return nocc, nspace, pmat_cutoff, scale_hcore, scale_vee

