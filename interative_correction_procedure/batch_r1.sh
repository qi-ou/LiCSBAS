unw_dir=`ls -d GEOCml* | head -n 1`
LiCSBAS11_check_unw.py -d $unw_dir -t TS_$unw_dir -u 0.6 -s
cp TS_$unw_dir/network/network11.png ${1}_network11.png
LiCSBAS120_choose_reference.py -d $unw_dir -t TS_$unw_dir
cp TS_$unw_dir/network/network120_remain_strong_cuts.png ${1}_network120_remain_strong_cuts.png
cp TS_$unw_dir/info/120_reference.png ${1}_120_reference.png
LiCSBAS130_sb_inv.py -l TS_$unw_dir/info/120strong_connected_links.txt --suffix 1 --inv_alg WLS --keep_incfile --nopngs
LiCSBAS131_residual_threshold.py --suffix 1
LiCSBAS132_3D_correction.py -o GEOCml10GACOS2 --suffix 1 -s 0.15
LiCSBAS_plot_network.py -i TS_$unw_dir/info/132good_ifg_uncorrected1_0.15_0.15.txt -b $unw_dir/baselines -s -m 3.4.5.6.7.8.9.10 -t 13
mv 132good_ifg_uncorrected1_0_dt_gt_13_months3.4.5.6.7.8.9.10_strongly_connected_network.png ${1}132good_ifg_uncorrected1_0_dt_gt_13_months3.4.5.6.7.8.9.10_strongly_connected_network.png
share ${1}132good_ifg_uncorrected1_0_dt_gt_13_months3.4.5.6.7.8.9.10_strongly_connected_network.png
mkdir -p TS_GEOCml10GACOS/130resid1/summer_uncorrected 
for i in `cat 132good_ifg_uncorrected1_0_dt_gt_13_months3.4.5.6.7.8.9.10_both_in.txt` ; do mv TS_GEOCml10GACOS/130resid1/good_ifg_no_correction/${i}.png TS_GEOCml10GACOS/130resid1/summer_uncorrected/ ; done 
mkdir -p TS_GEOCml10GACOS/130resid1/winter_uncorrected 
for i in `cat 132good_ifg_uncorrected1_0_dt_gt_13_months3.4.5.6.7.8.9.10_either_out.txt` ; do mv TS_GEOCml10GACOS/130resid1/good_ifg_no_correction/${i}.png TS_GEOCml10GACOS/130resid1/winter_uncorrected/ ; done 
