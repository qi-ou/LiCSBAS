#!/bin/bash
current_dir=`pwd`
frame=${current_dir##*/}
cp ~/bin/batch_final.sh .
cp ~/bin/sub_final.sh .
sed -i "s/final_\ /final_$frame\ /" sub_final.sh
unw_dir=`ls -d GEOCml* | head -n 1`
sed -i "s/GEOCml10GACOS/$unw_dir/g" batch_final.sh
./sub_final.sh $frame 
