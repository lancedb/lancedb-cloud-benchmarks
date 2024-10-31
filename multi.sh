if [ $# -lt 3 ]; then
    echo "Usage: $0 <number_of_processes> <prefix> <command> [processes_per_group]"
    echo "Note: processes_per_group defaults to 1 if not specified"
    exit 1
fi

n=$1
prefix=$2
cmd=$3
n_query_concurrency=${4:-1}

rm -f out.txt
for i in $(seq 1 $n); do
  for j in $(seq 1 $n_query_concurrency); do
    $cmd -p "$prefix-$i" & >> out.txt
  done
done

tail -f out.txt