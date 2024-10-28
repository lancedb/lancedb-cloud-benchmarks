n=$1
cmd=$2
rm -f out.txt
for i in $(seq 1 $n); do
  $cmd & >> out.txt
done

tail -f out.txt