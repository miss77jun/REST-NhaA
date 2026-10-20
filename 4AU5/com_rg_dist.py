import os

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis.base import analysis_class
from MDAnalysis.analysis import distances

import warnings

# suppress some MDAnalysis warnings about PSF files
warnings.filterwarnings('ignore')


def radgyr(atomgroup, masses, total_mass=None):
    # coordinates change for each frame
    coordinates = atomgroup.positions
    center_of_mass = atomgroup.center_of_mass()

    # get squared distance from center
    ri_sq = (coordinates - center_of_mass) ** 2
    # sum the unweighted positions
    sq = np.sum(ri_sq, axis=1)

    # make into array
    sq_rs = np.array([sq])

    # weight positions
    rog_sq = np.sum(masses * sq_rs, axis=1) / total_mass
    # square root and return
    return np.sqrt(rog_sq)


def batch_calc(model_name, end_dcd):
    RadiusOfGyration = analysis_class(radgyr)

    # f = open("lost_dcd.txt", "a")
    result_folder = './analysis_data/' + model_name + '/'
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    model_folder = "/media/huang/backupdisk02/nhaa/" + model_name + '/'
    psf_file = model_folder + "openmm/step5_input.psf"
    if not os.path.exists(psf_file):
        psf_file = model_folder + "openmm/step5_charmm2omm.psf"
    print(psf_file)

    df_rg = pd.DataFrame()
    df_com = pd.DataFrame()
    df_dist = pd.DataFrame()

    for j in range(1, end_dcd + 1):
        dcd = model_folder + 'openmm/' + 'step7_' + str(j) + '.dcd'
        if not os.path.exists(dcd):
            # f.write(model_name + "\tlost_dcd:" + str(dcd) + '\n')
            print(model_name + "\tlost_dcd:" + str(dcd))
            continue

        u = mda.Universe(psf_file, dcd)
        '''
        calculate radius of gyration
        '''
        # 向内
        atom_if = u.select_atoms(
            '(resid 134 or resid 157 or resid 160 or resid 161 or resid 75) and not backbone and not type H* and segid PROA')
        rog_u_if = RadiusOfGyration(u.trajectory, atom_if,
                                    atom_if.masses,
                                    total_mass=np.sum(atom_if.masses))
        rog_u_if.run()

        # 向外
        atom_of = u.select_atoms(
            "(resid 344 or resid 345 or resid 167 or resid 168 or resid 72) and not backbone and not type H* and segid PROA")
        rog_u_of = RadiusOfGyration(u.trajectory, atom_of,
                                    atom_of.masses,
                                    total_mass=np.sum(atom_of.masses))
        rog_u_of.run()

        if_col = rog_u_if.results['timeseries'][:, 0]  # if 取出'all'对应的列
        of_col = rog_u_of.results['timeseries'][:, 0]  # of 取出'all'对应的列
        df1 = pd.DataFrame({'if_rg': if_col, 'of_rg': of_col})
        df_rg = pd.concat([df_rg, df1], ignore_index=True)

        '''
        calculate center of mass
        '''
        # core
        core = u.select_atoms(
            "(resid 95 : 116 or resid 121 : 131 or resid 134 : 143 or resid 150 : 175 or resid 290 : 311 or resid 327 : 336 or resid 340 : 350 or resid 357 : 382) and segid PROA and backbone and not type H*")
        # dimer
        dimer = u.select_atoms(
            "(resid 12 : 30 or resid 182 : 200 or resid 59 : 85 or resid 205 : 218 or resid 223 : 236 or resid 247 : 271) and segid PROA and backbone and not type H*")

        '''
        calculate distance
        '''
        # select atom
        R81 = u.select_atoms('name CA and resid 81')
        L143 = u.select_atoms('name CA and resid 143')
        K300 = u.select_atoms('name NZ and resid 300')
        D163_1 = u.select_atoms('name OD1 and resid 163')
        D163_2 = u.select_atoms('name OD2 and resid 163')

        for ts in u.trajectory:
            core_com = core.center_of_mass()
            dimer_com = dimer.center_of_mass()
            df2 = pd.DataFrame({'x_core': [core_com[0]], 'y_core': [core_com[1]], 'z_core': [core_com[2]],
                                'x_dimer': [dimer_com[0]], 'y_dimer': [dimer_com[1]], 'z_dimer': [dimer_com[2]],
                                'x_delta': [core_com[0] - dimer_com[0]],
                                'y_delta': [core_com[1] - dimer_com[1]],
                                'z_delta': [core_com[2] - dimer_com[2]]})
            df_com = pd.concat([df_com, df2], ignore_index=True)

            resids1, resids2, dist1 = distances.dist(R81, L143, offset=0)
            resids3, resids4, dist2 = distances.dist(K300, D163_1, offset=0)
            resids5, resids6, dist3 = distances.dist(K300, D163_2, offset=0)
            if dist2[0] > dist3[0]:
                df3 = pd.DataFrame({'R81-L143': [dist1[0]], 'K300-D163': [dist3[0]]})
            else:
                df3 = pd.DataFrame({'R81-L143': [dist1[0]], 'K300-D163': [dist2[0]]})
            df_dist = pd.concat([df_dist, df3], ignore_index=True)

        '''
        save analysis result
        '''
        df_rg['sub'] = df_rg['if_rg'] - df_rg['of_rg']  # 相减操作并将结果存储在sub列中

        df_rg.to_csv(result_folder + model_name + '_rg' + '.csv')
        df_com.to_csv(result_folder + model_name + "_com" + ".csv")
        df_dist.to_csv(result_folder + model_name + "_dist" + ".csv")


def get_model_list(txt):
    f = open(txt, 'r')
    model_list = []
    lines = f.readlines()
    for line in lines:
        line = line.replace("\n", "")
        model_list.append(line)
    return model_list


def main():
    model_list = get_model_list("./model.txt")
    dcd_list = [279, 386, 241, 384, 158, 197, 150, 170, 8, 20, 14, 3, 3, 10, 3, 3, 69, 70]
    for i in range(len(model_list)):
        print(model_list[i] + "\tend_dcd:" + str(dcd_list[i]))
        batch_calc(model_list[i], dcd_list[i])

    # batch_calc("1zcd-aa-k300n", 386)


if __name__ == '__main__':
    main()
