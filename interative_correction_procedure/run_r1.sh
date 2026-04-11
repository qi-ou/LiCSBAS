#!/bin/bash
current_dir=`pwd`
frame=${current_dir##*/}
cp /gws/nopw/j04/nceo_geohazards_vol2/LiCS/temp/insar_proc/earqo/tianshan/use_qi_approach/batch_r1.sh .
cp /gws/nopw/j04/nceo_geohazards_vol2/LiCS/temp/insar_proc/earqo/tianshan/use_qi_approach/sub_r1.sh .
sed -i "s/r1_\ /r1_$frame\ /" sub_r1.sh
unw_dir=`ls -d GEOCml* | head -n 1`
sed -i "s/GEOCml10GACOS/$unw_dir/g" batch_r1.sh
./sub_r1.sh $frame
