# Improvement and current problems
# Quality
+ Undistortion - com between undistorted images and distorted images
+ Editablity
+ Limited Point number - you can increase the --densify_grad_threshold, --densification_interval or reduce the value of --densify_until_iter. Note however that this will affect the quality of the result. Also try setting --test_iterations to -1 to avoid memory spikes during testing
+ sp
# Performance
+ gradiant operations?
+ radii - check if sparse?
  + sparse
  + but mainly cuda synchronize problem.