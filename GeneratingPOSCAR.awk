#awk '/POSITION/ {for(i=1; i<=288;i++) {getline; print $1,$2,$3}}' inputfile

#awk '/POSITION/ {for(i=1; i<=288;i++) {getline; print $1,$2,$3}}' inputfile | tail -287

