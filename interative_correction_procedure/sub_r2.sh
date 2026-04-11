bsub2slurm.sh -o r2.out -e r2.err -J r2_ -n 4 -W 10:00 -M 16000 -q comet ./batch_r2.sh $1
