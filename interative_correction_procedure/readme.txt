This folder contains batch files that are used to repeat Qi Ou's LiCSBAS workflow that involves interations of time series analysis, automatic unwrapping mistake correction, and sorting corrected interferograms for visual checking, and running time series inversion again. First add this folder to your $PATH, then:

1. run Milan's licsar2licsbas for a frame
licsar2licsbas.sh -M 10 -P -n 8 -T -g -W -u $frameID

2. go into a frame directory and type:
run_r1.sh
This will run step 11-132, and complete the first iteration of time series analysis, unwrapping correction, and sorting interferograms based on whether and how they are corrected, and by seasons defined by months, which you can adapt in the batch_r1.sh file. 

3. Download to local the folders containing pngs under the TS*/130resid1/\*corr\* for visual checking.

4. Sorting pngs into bad and good with visual checking

5. Compile a list of good pngs called final_list.txt on Jasmin for the second interation. 

6. run_final.sh
If it's a good frame, the masking png will show almost zero loop errors after this step. If there are only few pixels with errors, happily type run_r2.sh to let it mask the bad pixels in those affected interferograms and run the time series inversion again. The index increases to keep the corrected ifgs separate from the original. So you can do run_r3.sh or more as needed. 

7. LiCSBAS_cum2vel_weighted.py -i cum.h5 -p -l -s --heading -10
with heading = -10 for ascending and heading = -170 for descending. 
This will produce a better velocity and vstd map that's less affected by the long-wavelength oscillations of the time series.
 


