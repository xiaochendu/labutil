# %%
import os
import sys
sys.path.append("/home/dux/")

# set env var
os.environ["LAMMPS_POTENTIALS"] = "/home/pleon/mylammps/potentials/"
os.environ["LAMMPS_COMMAND"] = "/home/pleon/mylammps/src/lmp_serial"
os.environ['WORKDIR'] = "/home/dux/3_220/"

from labutil.src.plugins.lammps import (lammps_run, get_lammps_energy)
from labutil.src.objects import (ClassicalPotential, Struc, ase2struc, Dir)
from ase.spacegroup import crystal
# from ase.build import *
import numpy as np
import matplotlib.pyplot as plt

# %%
eam_input_template = """
# ---------- 1. Initialize simulation ---------------------
units metal
atom_style atomic
dimension  3
boundary   p p p
read_data $DATAINPUT

# ---------- 2. Specify interatomic potential ---------------------
pair_style eam
pair_coeff * * $POTENTIAL

# pair_style lj/cut 4.5
# pair_coeff 1 1 0.3450 2.6244 4.5

# ---------- 3. Run single point calculation  ---------------------
thermo_style custom step pe lx ly lz press pxx pyy pzz
run 0

# ---- 4. Define and print useful variables -------------
variable natoms equal "count(all)"
variable totenergy equal "pe"
variable length equal "lx"

print "Total energy (eV) = ${totenergy}"
print "Number of atoms = ${natoms}"
print "Lattice constant (Angstoms) = ${length}"
        """

lj_input_template = """
# ---------- 1. Initialize simulation ---------------------
units metal
atom_style atomic
dimension  3
boundary   p p p
read_data $DATAINPUT

# ---------- 2. Specify interatomic potential ---------------------
# pair_style eam
# pair_coeff * * $POTENTIAL

pair_style lj/cut 4.5
pair_coeff 1 1 0.3450 2.6244 4.5

# ---------- 3. Run single point calculation  ---------------------
thermo_style custom step pe lx ly lz press pxx pyy pzz
run 0

# ---- 4. Define and print useful variables -------------
variable natoms equal "count(all)"
variable totenergy equal "pe"
variable length equal "lx"

print "Total energy (eV) = ${totenergy}"
print "Number of atoms = ${natoms}"
print "Lattice constant (Angstoms) = ${length}"
        """
# %%
def make_struc(alat):
    """
    Creates the crystal structure using ASE.
    :param alat: Lattice parameter in angstrom
    :return: structure object converted from ase
    """
    unitcell = crystal('Ag', [(0, 0, 0)], spacegroup=225, cellpar=[alat, alat, alat, 90, 90, 90])
    #multiplier = numpy.identity(3) * 2
    #ase_supercell = make_supercell(unitcell, multiplier)
    structure = Struc(ase2struc(unitcell))
    return structure


def compute_energy(alat, pot_type, template):
    """
    Make an input template and select potential and structure, and the path where to run
    """
    potpath = os.path.join(os.environ['LAMMPS_POTENTIALS'],'Ag_u3.eam')
    if pot_type == "eam":
        potential = ClassicalPotential(path=potpath, ptype='eam', element=["Ag"])
    else:
        potential = ""
    runpath = Dir(path=os.path.join(os.environ['WORKDIR'], "Lab1", pot_type+"_"+str(alat)))
    struc = make_struc(alat=alat)
    output_file = lammps_run(struc=struc, runpath=runpath, potential=potential, intemplate=template, inparam={})
    energy, lattice = get_lammps_energy(outfile=output_file)
    return energy, lattice


def lattice_scan(pot):
    alat_list = np.linspace(3.8, 4.3, 7)
    if pot == "eam":
        energy_list = [compute_energy(alat=a, pot=pot, template=eam_input_template)[0] for a in alat_list]
    else:
        energy_list = [compute_energy(alat=a, pot=pot, template=lj_input_template)[0] for a in alat_list]

    print(energy_list)
    plt.plot(alat_list, energy_list)
    plt.savefig("Ag_lat_" + pot)
    plt.show()


# if __name__ == '__main__':
# %%
# put here the function that you actually want to run
potentials = ["lj", "eam"]
[lattice_scan(pot) for pot in potentials]
