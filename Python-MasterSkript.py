import os

def POS(file_no,position):
    with open(f"POSCAR_{file_no}","w") as pos:
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

file_no = -1
position = []

os.system("awk '/POSITION/ {for(i=1; i<=288; i++) {getline; print $1,$2,$3}}' OUTCAR > all_pos.dat")

with open('all_pos.dat','r') as pos:
    for line in pos:
        line = line.rstrip('\n')
        if line == '-----------------------------------------------------------------------------------':
            if file_no != -1:
                POS(file_no,position)                 
            position = []
            file_no += 1
        elif line == '  ':
            pass
        else:
            position.append(line)

for N in range(file_no):
    for atom in range(1, 97): #just the Ti Atoms
        os.system("chmod +x neighbors.pl")
        os.system(f"./neighbors.pl POSCAR_{N} {atom}")
        with open("neighdist.dat") as dis:
            distances = [next(dis) for x in range(9)]
            with open(f"Distances_POS{N}_ATOM{atom}.dat","w")as dat:
                dat.write(distances)