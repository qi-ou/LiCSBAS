bsub2slurm.sh -o final.out -e final.err -J final_ -n 4 -W 10:00 -M 16000 -q comet ./batch_final.sh $1
