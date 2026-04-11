LiCSBAS130_sb_inv.py --inv_alg WLS --keep_incfile --nopngs -l final_list.txt --suffix 2 
LiCSBAS131_residual_threshold.py --suffix 2
LiCSBAS133_write_h5.py -l final_list.txt --suffix 2
LiCSBAS14_vel_std.py -t TS_GEOCml10GACOS
LiCSBAS15_mask_ts.py -t TS_GEOCml10GACOS -c 0.05 -u 1.5 -v 2 -T 3 -g 5 -s 5 -i 200 -l 5 -r 1.5
cp TS_GEOCml10GACOS/results/mask_ts.png ${1}_new_mask_ts.png
