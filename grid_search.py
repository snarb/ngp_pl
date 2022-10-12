import subprocess
import os
# [76, 70, 64, 56, 42, 36, 28, 16, 12, 8, 4, 2, 1] # 29, 29, 29
#['1e-1', '1e-2', '6e-3',  '3e-3', '1e-3', '5e-4', '1e-4'
for cur_scale in [34]:
    print('----------------------------------------')
    print("Scale: ", str(cur_scale))
    for dl in ['6e-3', '5e-3', '4e-3', '3e-3', '25e-3', '2e-3', '1e-3']: # 0.003 3e-3 Looks like the best

        for _ in range(3):
            print('----------------------------------------')
            print("DL: ", str(dl))
            p = subprocess.Popen(
                "/home/ubuntu/anaconda3/envs/ngp_pl2/bin/python3 /home/ubuntu/repos/ngp_pl/train_orig.py --root_dir /home/ubuntu/repos/instant-ngp-flame/ "
                "--dataset_name colmap --exp_name scales_no_find_dl   --num_gpus 1  --num_epochs 5 --downsample 0.5 --batch_size 10000 --distortion_loss_w  " + str(dl)  + ' '#--optimize_ext   --random_bg  
                "--lr 1e-2 --scale " + str(cur_scale), shell=True)
            (output, err) = p.communicate()

            # This makes the wait possible
            p_status = p.wait()
            if output is not None:
                print ("Command output: " + output)
            if err is not None:
                print ("Error: " + err)
            print('----------------------------------------')
