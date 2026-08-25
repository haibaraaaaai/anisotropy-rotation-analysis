Background percentage heatmaps give background percentage or rod signal strength. r dependence enters through I(theta) relation - I used a fitted intensity(r) profile from datasets of stuck rods on glass previously used for precision calculations shown in meetings.

From largest to smallest mean background:
- uncorrected_background: frame_stack_20260716-140000 average frame.npy
- different_coverslip_background_subtracted: frame_stack_20260716-140137 average frame.npy
- same_point_same_coverslip_best_case_background_subtracted: frame_stack_20260716-140047 average frame.npy

Each rod size has one heatmap CSV and PNG per condition, giving 6 CSVs and 6 PNGs.
CSV format: first column is percentage_bin_center; each following column is an r value; cells are probability density.
No additional subtraction is applied to the stored average-frame npy files. 

Input folder: C:\Polarcam Software\Polarcam_v3\polarcam_live\background characterisation\16072026 backgrounds
Output folder: C:\Polarcam Software\Polarcam_v3\polarcam_live\background characterisation\16072026 backgrounds\labelled percentage heatmaps no extra subtraction



