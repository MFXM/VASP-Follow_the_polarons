import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})


#-------------- Creating the directorys-----------------#
path = ["Diagramms", "Avg_Distance", "Avg_Distnce_with_Polaron","Variance", "Variance_with_Polaron"]

for p in path:
    try:
        os.mkdir(p)
    except OSError:
        print ("Directory \"%s\" allready exists" % p)
    else:
        print ("creating Directory \"%s\"" % p)

def op(path,d):
    with open(path) as dat:
        for line in dat:
            d.append(float(line))


#--------------- Printing the distances-----------------#
print("Creating Plots and Avereg of Distance...")
for t in range(1,97):
    
    
    #---------Calculating Avg and Var------------------#
    
    awk1 = """awk 'BEGINN{cnt=0} { for(i=1;i<=NF;i++) {if($9 < 0.5 && $9 > -0.5) { sum[i]+=$i ; sq[i]+=$i*$i ; }} } {if($9 < 0.5 && $9 > -0.5) ++cnt} 
              END {{if(cnt != 0)
                   for(i=1;i<=NF;i++) printf sum[i]/cnt " " ;
                   for(i=1;i<=NF;i++) printf sq[i]/cnt-(sum[i]/cnt)**2 " ";
                  }}' Ti_%s.dat > temp.dat """ %t
    os.system(awk1)
    
    awk2 = "awk '{for(i=1;i<=9;i++) print $i}' temp.dat > Avg_Distance/Avg_Distance_Ti_%s.dat" %t
    os.system(awk2)
    awk3 = "awk '{for(i=10;i<=18;i++) print $i}' temp.dat > Variance/Variance_Ti_%s.dat" %t
    os.system(awk3)
   
    os.remove("temp.dat")
              
    awk4 = """awk 'BEGINN{cnt=0} { for(i=1;i<=NF;i++) {if($9 > 0.5 || $9 < -0.5) { sum[i]+=$i ; sq[i]+=$i*$i ; }} } {if($9 > 0.5 || $9 < -0.5) ++cnt} 
              END {
                  {if(cnt != 0){
                   for(i=1;i<=NF;i++) printf sum[i]/cnt " ";
                   for(i=1;i<=NF;i++) printf sq[i]/cnt-(sum[i]/cnt)**2 " " ;}
                  }}' Ti_%s.dat > temp.dat""" %t          
    os.system(awk4)
    
    awk5 = "awk '{for(i=1;i<=9;i++) print $i}' temp.dat > Avg_Distnce_with_Polaron/Avg_Distance_with_Polaron_Ti_%s.dat" %t
    os.system(awk5)
    awk6 = "awk '{for(i=10;i<=18;i++) print $i}' temp.dat > Variance_with_Polaron/Variance_with_Polaron_Ti_%s.dat" %t
    os.system(awk6)
    
    rm = []
    op(f"Avg_Distnce_with_Polaron/Avg_Distance_with_Polaron_Ti_{t}.dat",rm)
    if(len(rm) == 0):
        os.remove(f"Avg_Distnce_with_Polaron/Avg_Distance_with_Polaron_Ti_{t}.dat")
    
    rm = []
    op(f"Variance_with_Polaron/Variance_with_Polaron_Ti_{t}.dat",rm)
    if(len(rm) == 0):
        os.remove(f"Variance_with_Polaron/Variance_with_Polaron_Ti_{t}.dat")
        
    #-------------------Creating the plots-----------------------#
    awk = "awk '{print ($7 + $8)/2}' Ti_%s.dat > temp.dat" %t
    os.system(awk)
    dist = []
    op("temp.dat",dist)
    os.remove("temp.dat")
    
    plt.figure(t)
    plt.plot(dist)
    plt.ylabel("Ti- Ti Distance [A]"); plt.xlabel("Time [ns]")
    plt.title(f"MD of the {t}th Ti-Atom")
    plt.grid(True)
    plt.savefig(f"Diagramms/A-Ti-Ti_Distance_Ti_{t}.png")
