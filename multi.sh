set -x
n=$1
prefix=$2
cmd=$3
rm -f out.txt
for i in $(seq 1 $n); do
  $cmd -p "$prefix-$i" & >> out.txt
done

tail -f out.txt