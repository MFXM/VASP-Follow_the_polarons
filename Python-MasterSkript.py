import os
import numpy as np

def POS(file_no,position):
    with open(f"POSCAR/POSCAR_{file_no}","w") as pos:
        pos.write(f"Automatic Generated POSCAR file number : {file_no}\n")
        pos.write("1\n")
        pos.write("12.9730319976999997    0.0000000000000000    0.0000000000000000\n")
        pos.write("0.0000000000000000   12.9730319976999997    0.0000000000000000\n")
        pos.write("0.0000000000000000    0.0000000000000000   17.7244205474999994\n")
        pos.write("Ti O\n")
        pos.write("96 192\n")
        pos.write("Cartesian\n")
        for i in range(len(position)):
            pos.write(position[i])
            pos.write("\n")
            
def MAG(file_no,magnet):
    with open(f"MAGNETIZATION/MAGNETIZATION_{file_no}","w") as mag:
        for i in range(len(magnet)):
            mag.write(magnet[i])
            mag.write("\n")

file_no = -1
file_no2 = 0
position = []
magnet = []


os.system("awk '/POSITION/ {for(i=1; i<=289; i++) {getline; print $1,$2,$3}}' OUTCAR > all_pos.dat")

with open('all_pos.dat') as pos:
    for line in pos:
        line = line.rstrip('\n')
        if line == '-----------------------------------------------------------------------------------  ':
            if file_no != -1:
                POS(file_no,position)                 
            position = []
            file_no += 1
        else:
            position.append(line)
            
POS(file_no + 1, position)

file_no = 1

os.system("awk 'BEGIN{t=0; m=99999999} $1 ~ /magnetization/ && $2 ~ /x/ {m=NR; t=t+1} {if(NR > m+3 && NR < m+292) print $5 } {if(NR == m+292) print \"-----\" }' OUTCAR > MAGNETIZATION/all_mag.dat")       
#os.system("awk 'BEGIN{t=0; m=99999999} $1 ~ /magnetization/ && $2 ~ /x/ {m=NR; t=t+1} {if(NR > m+3 && NR < m+292) print $5 > \"MAGNETIZATION/MAGNETIZATION_\"t } ' OUTCAR")

with open('MAGNETIZATION/all_mag.dat') as mag:
    for line in mag:
        line = line.rstrip('\n')
        if line == '-----':
            MAG(file_no2,magnet)                 
            magnet = []
            file_no2 += 1
        else:
            magnet.append(line)

for N in range(file_no):
    with open(f"MAGNETIZATION/MAGNETIZATION_{N}") as mag:
        magnetization = [next(mag) for x in range(288)]
    for atom in range(1, 2): #just the 96 Ti Atoms
        os.system("chmod +x neighbors.pl")
        os.system(f"./neighbors.pl POSCAR/POSCAR_{N} {atom}")
        with open("neighdist.dat") as dis:
            distances = [next(dis) for x in range(9)]
        with open("Distances.dat","w")as dat:
            for i in range(len(distances)):
                dat.write(distances[i])
        
        os.system("awk '{print $2}' Distances.dat > atom.dat")
        os.system("awk '{print $NF}' Distances.dat > dis.dat")
        with open('atom.dat') as distance:
            real_atom = [next(distance) for x in range(9)]
        with open('dis.dat') as distance:
            real_distance = [next(distance) for x in range(9)]
            
        datafile = ''
        
        with open(f"DATA/Ti_{atom}.dat","a+") as data:
            for i in range(1,9): 
                datafile = datafile + real_distance[i].strip('\n') + ' '
            datafile = datafile + magnetization[i].strip('\n') + ''
            datafile = datafile + str(N)
            
            data.write(datafile)
        
        with open(f"DATA_control/Ti_{atom}_control.dat","a+") as data:
            for i in range(1,9): 
                datafile = datafile + real_atom[i].strip('\n') + ' '
            datafile = datafile + magnetization[i].strip('\n') + ' '
            datafile = datafile + str(N)
            
            data.write(datafile)
        
        