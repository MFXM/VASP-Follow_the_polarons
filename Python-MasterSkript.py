import os


# Creates POSCAR File from the position
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
            
# Creates MAGNETIZATION File 
def MAG(file_no,magnet):
    with open(f"MAGNETIZATION/MAGNETIZATION_{file_no}","w") as mag:
        for i in range(len(magnet)):
            mag.write(magnet[i])
            mag.write("\n")

file_no = -1
file_no2 = 0
position = []
magnet = []

print("Preparing data ...")

# Graps position information from OUTCAR Files
os.system("awk '/POSITION/ {for(i=1; i<=289; i++) {getline; print $1,$2,$3}}' OUTCAR > all_pos.dat")


# Read position information
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

# Wirte POSCAR Files            
POS(file_no + 1, position)

# Graps magnetization information from OUTCAR File
os.system("awk 'BEGIN{t=0; m=99999999} $1 ~ /magnetization/ && $2 ~ /x/ {m=NR; t=t+1} {if(NR > m+3 && NR < m+292) print $5 } {if(NR == m+292) print \"-----\" }' OUTCAR > MAGNETIZATION/all_mag.dat")       
#os.system("awk 'BEGIN{t=0; m=99999999} $1 ~ /magnetization/ && $2 ~ /x/ {m=NR; t=t+1; print > \"MAGNETIZATION/MAGNETIZATION_\"t} {if(NR > m+3 && NR < m+292) print $5 > \"MAGNETIZATION/MAGNETIZATION_\"t } ' OUTCAR")

# Read magnetization information 
with open('MAGNETIZATION/all_mag.dat') as mag:
    for line in mag:
        line = line.rstrip('\n')
        if line == '-----':
            MAG(file_no2,magnet)                 
            magnet = []
            file_no2 += 1
        else:
            magnet.append(line)
            
            
# Prepares atom files with all relevant information (labels)        
for atom in range(1,97):     
    label = ''
    with open(f"DATA_control/Ti_{atom}_control.dat","w+") as data:
        for i in range(1,9): 
            label = label + 'POSITION '
        label = label + 'Magnetization '
        for i in range(1,9): 
            label = label + 'ATOM '
        label = label + 'Time (fs) \n\n'
        
        data.write(label)

#Evaluates information
for N in range(file_no):
    if N%100 == 0:
        print(f"Evaluating data at t = {N}fs ...")
    #Reading magnetization from MAGNETIZATION Files
    with open(f"MAGNETIZATION/MAGNETIZATION_{N}") as mag:
        magnetization = [next(mag) for x in range(288)]
        
    # Using the neigbours script evaluating the distances
    for atom in range(1, 97): #just the 96 Ti Atoms
        os.system("chmod +x neighbors.pl")
        os.system(f"./neighbors.pl POSCAR/POSCAR_{N} {atom}")
        with open("neighdist.dat") as dis:
            distances = [next(dis) for x in range(9)]
        with open("Distances.dat","w")as dat:
            for i in range(len(distances)):
                dat.write(distances[i])
        
        #extracting atom information and distances
        os.system("awk '{print $2}' Distances.dat > atom.dat")
        os.system("awk '{print $NF}' Distances.dat > dis.dat")
        with open('atom.dat') as distance:
            real_atom = [next(distance) for x in range(9)]
        with open('dis.dat') as distance:
            real_distance = [next(distance) for x in range(9)]
            
        datafile = ''
        
        # Saving distance information
        with open(f"DATA/Ti_{atom}.dat","a+") as data:
            for i in range(1,9): 
                datafile = datafile + real_distance[i].strip('\n') + ' '
            datafile = datafile + magnetization[atom-1].strip('\n') + '\n'
            
            data.write(datafile)
        
        # saving atom information 
        with open(f"DATA_control/Ti_{atom}_control.dat","a+") as data:
            datafile = datafile.strip('\n')
            for i in range(1,9): 
                datafile = datafile + real_atom[i].strip('\n') + ' '
            datafile = datafile + str(N) + '\n'
            
            data.write(datafile)
        