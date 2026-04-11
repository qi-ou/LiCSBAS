#!/bin/bash
current_dir=`pwd`
frame=${current_dir##*/}
cp ~/bin/batch_r2.sh .
cp ~/bin/sub_r2.sh .
sed -i "s/r2_\ /r2_$frame\ /" sub_r2.sh
unw_dir=`ls -d GEOCml* | head -n 1`
sed -i "s/GEOCml10GACOS/$unw_dir/g" batch_r2.sh
./sub_r2.sh $frame 
