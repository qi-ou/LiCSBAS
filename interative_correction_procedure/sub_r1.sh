bsub2slurm.sh -o r1.out -e r1.err -J r1_ -n 4 -W 10:00 -M 16000 -q comet ./batch_r1.sh $1
