#coding=utf-8
import numpy as np
from tqdm import tqdm


class MATCH(object):
    def __init__(self,workpath,file_name):
        self.internal_ions_maxlength=2
        self.workpath = workpath
        self.file_name = file_name
        self.amino_acid={'A':71.037114,'C':103.009185,'D':115.026943,'E':129.042593,'F':147.068414,\
        'G':57.021464,'H':137.058912,'I':113.084064,'K':128.094963,'L':113.084064,\
        'M':131.040485,'N':114.042927,'P':97.052764,'Q':128.058578,'R':156.101111,\
        'S':87.032028,'T':101.047678,'V':99.068414,'W':186.079313,'Y':163.063329,\
        'c':160.0306486796,'m':147.035399708}

        self.PROTON= 1.007276466583
        self.H = 1.0078250322
        self.O = 15.9949146221
        self.CO = 27.9949146200
        self.N = 14.0030740052
        self.C = 12.00
        self.isotope = 1.003

        self.CO = self.C + self.O
        self.CO2 = self.C + self.O * 2
        self.NH = self.N + self.H
        self.NH3 = self.N + self.H * 3
        self.HO = self.H + self.O
        self.H2O= self.H * 2 + self.O

    def read_mgf(self):
        mgf_content = []
        with open(self.workpath + '/' + self.file_name, 'r') as rf:
            while True:
                line = rf.readline()
                if not line:
                    break
                if 'BEGIN IONS' in line:
                    _list=[]
                    title = rf.readline().split('=')[1].replace('\n','')
                    Sequnce = rf.readline().split('=')[1].replace("\n", "")
                    if Sequnce.startswith('DECOY-'):
                        Sequnce = Sequnce.split('-')[1]
                    Charge=rf.readline().split('=')[1].replace("\n", "")
                    Modified = rf.readline().split('=')[1].replace("\n", "")
                    Sequnce = self.flush_Sequence(Sequnce, Modified)
                    rf.__next__();rf.__next__()
                    line = rf.readline()
                    temp_list = []
                    while 'END IONS' not in line:
                        temp_ = []
                        temp_.append(float(line.split()[0]))
                        temp_.append(float(line.split()[1]))
                        temp_list.append(temp_)
                        line = rf.readline()
                    _list.append(Sequnce)
                    _list.append(Charge)
                    _list.append(temp_list)
                    _list.append(title)
                    mgf_content.append(_list)
        print('Read mgf successful! ' + self.file_name)
        return mgf_content  ##[[sequence1,charge1,[[mz11,intensity11],[mz12,intensity12]],spectrum_title],[sequence2,charge2,[[mz21,intensity21],[mz22,intensity22]],nce,spectrum_title]]

    def get_mass(self,Amino_acid):
        return self.amino_acid[Amino_acid]

    def flush_Sequence(self,Sequence,Modified):
        Modified=Modified.replace(' ','').split(';')
        for i in Modified:
            if i:
                if 'Oxidation@M' in i:
                    temp_Sequence = list(Sequence)
                    temp_Sequence[int(i[11:])-1]= 'm'
                    Sequence=''.join(temp_Sequence)
                if 'Carbamidomethyl@C' in i:
                    Sequence = Sequence.replace('C','c')
        return Sequence

    def get_by_mz(self,sequence,charge):
        charge = int(charge)
        peptide_list = list(sequence)
        b_ions = []
        y_ions = []
        for i in range(len(sequence)):
            b_ions.append(sequence[:i])
            y_ions.append(sequence[len(sequence)-i:])
        peptide_list_mass = list(map(self.get_mass,peptide_list))
        b_mass_list = np.divide(np.cumsum(np.array(peptide_list_mass))+self.PROTON*charge,charge)[:-1]
        y_mass_list = np.divide(np.cumsum(np.array(peptide_list_mass[::-1]))+self.PROTON*charge+self.H2O,charge)[:-1]
        a_mass_list = np.divide(np.cumsum(np.array(peptide_list_mass))+self.PROTON*charge-self.CO,charge)[:-1]
        return b_ions[1:],b_mass_list.tolist(),y_ions[1:],y_mass_list.tolist(),a_mass_list

    def get_ions_mz(self,sequence):
        b_ions1,b_mass_list1,y_ions1,y_mass_list1,a_mass_list1 = self.get_by_mz(sequence,1)
        b_ions2,b_mass_list2,y_ions2,y_mass_list2,a_mass_list2 = self.get_by_mz(sequence,2)
        y_ions1.reverse()
        y_mass_list1.reverse()
        y_ions2.reverse()
        y_mass_list2.reverse()
        return b_ions1,b_mass_list1,y_ions1,y_mass_list1,b_ions2,b_mass_list2,y_ions2,y_mass_list2,a_mass_list1,a_mass_list2

    def match_ions(self):
        mgf = self.read_mgf()
        ions_results = []
        ions_error = []
        print('start Annotated ions :')
        for i in tqdm(range(len(mgf))):
            temp_spectrum = mgf[i]
            mz_intensity = temp_spectrum[2]
            spectrum_title = temp_spectrum[3]
            ions_mz = self.get_ions_mz(temp_spectrum[0])
            bs_ions1 = np.array(ions_mz[1],dtype=np.float64)
            ys_ions1 = np.array(ions_mz[3],dtype=np.float64)
            bs_ions2 = np.array(ions_mz[5],dtype=np.float64)
            ys_ions2 = np.array(ions_mz[7],dtype=np.float64)
            as_ions1 = np.array(ions_mz[8],dtype=np.float64)
            as_ions2 = np.array(ions_mz[9],dtype=np.float64)
            np_mz_intensity = np.array(mz_intensity,dtype=np.float64)
            mz = np_mz_intensity[:,0]
            max_intensity = np.max(np_mz_intensity[:,1].astype(float))
            for index in range(len(bs_ions1)):              ##regular ions
                ions_results.append(temp_spectrum[0])
                ions_results.append(temp_spectrum[1])
                matched_mz = []
                intensity = []
                ion_type = ['b'+str(index+1)+'+','b'+str(index+1)+'++','y'+str(len(bs_ions1)-index)+'+','y'+str(len(bs_ions1)-index)+'++']
                dif_a1 = ((mz-as_ions1[index])/as_ions1[index])*10**6
                try:
                    if np.min(abs(dif_a1))<20:
                        matched_index = np.where(np_mz_intensity == np.max(np_mz_intensity[np.where(abs(dif_a1) < 20)]))[0]
                        np_mz_intensity=np.delete(np_mz_intensity,matched_index,0)
                        mz = np_mz_intensity[:,0]
                except:
                    pass
                dif_a2 = ((mz-as_ions2[index])/as_ions2[index])*10**6
                try:
                    if np.min(abs(dif_a2))<20:
                        matched_index = np.where(np_mz_intensity == np.max(np_mz_intensity[np.where(abs(dif_a2) < 20)]))[0]
                        np_mz_intensity=np.delete(np_mz_intensity,matched_index,0)
                        mz = np_mz_intensity[:,0]
                except:
                    pass
                dif_b1 = ((mz-bs_ions1[index])/bs_ions1[index])*10**6
                ions_results.append(','.join([ions_mz[0][index],ions_mz[2][index]]))
                ions_results.append(','.join(ion_type))
                try:
                    if np.min(abs(dif_b1))<20:
                        matched_index = np.where(np_mz_intensity == np.max(np_mz_intensity[np.where(abs(dif_b1) < 20)]))[0]
                        matched_mz.append(str(mz[matched_index][0]))
                        intensity.append(str(np_mz_intensity[matched_index,1][0]))
                        ions_error.append([str(mz[matched_index][0]),str(dif_b1[matched_index][0]),str((np_mz_intensity[:,1][matched_index][0])/max_intensity)])
                        np_mz_intensity=np.delete(np_mz_intensity,matched_index,0)
                        mz = np_mz_intensity[:,0]
                    else:
                        matched_mz.append('0.0')
                        intensity.append('0.0')
                except:
                    matched_mz.append('0.0')
                    intensity.append('0.0')
                dif_b2 = ((mz-bs_ions2[index])/bs_ions2[index])*10**6
                try:
                    if np.min(abs(dif_b2))<20:
                        matched_index = np.where(np_mz_intensity == np.max(np_mz_intensity[np.where(abs(dif_b2) < 20)]))[0]
                        matched_mz.append(str(mz[matched_index][0]))
                        intensity.append(str(np_mz_intensity[matched_index,1][0]))
                        ions_error.append([str(mz[matched_index][0]),str(dif_b2[matched_index][0]),str((np_mz_intensity[:,1][matched_index][0])/max_intensity)])
                        np_mz_intensity=np.delete(np_mz_intensity,matched_index,0)
                        mz = np_mz_intensity[:,0]
                    else:
                        matched_mz.append('0.0')
                        intensity.append('0.0')
                except:
                    matched_mz.append('0.0')
                    intensity.append('0.0')
                dif_y1 = ((mz-ys_ions1[index])/ys_ions1[index])*10**6
                try:
                    if np.min(abs(dif_y1))<20:
                        matched_index = np.where(np_mz_intensity == np.max(np_mz_intensity[np.where(abs(dif_y1) < 20)]))[0]
                        matched_mz.append(str(mz[matched_index][0]))
                        intensity.append(str(np_mz_intensity[matched_index,1][0]))
                        ions_error.append([str(mz[matched_index][0]),str(dif_y1[matched_index][0]),str((np_mz_intensity[:,1][matched_index][0])/max_intensity)])
                        np_mz_intensity=np.delete(np_mz_intensity,matched_index,0)
                        mz = np_mz_intensity[:,0]
                    else:
                        matched_mz.append('0.0')
                        intensity.append('0.0')
                except:
                    matched_mz.append('0.0')
                    intensity.append('0.0')
                dif_y2 = ((mz-ys_ions2[index])/ys_ions2[index])*10**6
                try:
                    if np.min(abs(dif_y2))<20:
                        matched_index = np.where(np_mz_intensity == np.max(np_mz_intensity[np.where(abs(dif_y2) < 20)]))[0]
                        matched_mz.append(str(mz[matched_index][0]))
                        intensity.append(str(np_mz_intensity[matched_index,1][0]))
                        ions_error.append([str(mz[matched_index][0]),str(dif_y2[matched_index][0]),str((np_mz_intensity[:,1][matched_index][0])/max_intensity)])
                        np_mz_intensity=np.delete(np_mz_intensity,matched_index,0)
                        mz = np_mz_intensity[:,0]
                    else:
                        matched_mz.append('0.0')
                        intensity.append('0.0')
                except:
                    matched_mz.append('0.0')
                    intensity.append('0.0')
                ions_results.append(','.join(matched_mz))
                norm_intensity = map(str,(np.array(intensity,dtype=np.float64)/max_intensity).tolist())
                ions_results.append(','.join(norm_intensity))
                ions_results.append(','.join(intensity))
                ions_results.append(temp_spectrum[3])
                ions_results.append(spectrum_title)
        return ions_results
            # ions_results:[Sequence,charge,ions_sequence,ions_type,ions_mz,ions_norm_intensity,ions_intensity,nce,spectrum_title]

    def write_files(self):
        ions_results = self.match_ions()
        start = 0
        while True:
            out_file_name = self.file_name.split('.')[0] + '_byions'+'.txt'
            with open(self.workpath + '/' + out_file_name,'a+') as t:
                line = ions_results[start:start+7]
                line.insert(3,ions_results[start+8])
                line = '\t'.join(line)+'\n'
                t.write(line)
                start += 9
            if start+7 > len(ions_results):
                break
        print('get b_y ions successed!')