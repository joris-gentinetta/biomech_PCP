# Upper Limb Controller Study

### EMG channels: 

- [emg, '0'] # 0
- [emg, '1'] # 1
- [emg, '2'] # 2
- [emg, '4'] # 3
- [emg, '10'] # 4
- [emg, '11'] # 5
- [emg, '12'] # 6
- [emg, '13'] # 7

### Movement Order:

- 'thumbFlEx',
- 'thumbAbAd',
- 'indexFlEx',
- 'mrpFlEx',
- 'fingersFlEx',
- 'handOpCl',
- 'pinchOpCl',
- 'pointOpCl',
- 'keyOpCl',
- 'indexFlDigitsEx',
- 'wristFlEx',
- 'wristFlHandCl'

## Preaparation
Copy the <person_ID> folder, rename it with the actual person_ID. Adapt the <person_ID> in this file and in the name field of all the config files.

## Recording Data
Ask the participant to start the movement, then start the recording (makes cropping easier). After the first recording, check the data. There is no in plane calibration for the first 10 recordings (no wrist), record for 70 seconds. For the last two (wrist) recordings, ask them to move in plane, start the recording and after 5 seconds move to forward orientation of the lower arms / start movement. Record for 80 seconds total.
```
Check data:
 python s3_process_video.py --data_dir data/<person_ID>/recordings/thumbFlEx --experiment_name 1
```
```
Record data:
  python s7_online_learning.py --save_input --person_dir <person_ID> --experiment_name thumbFlEx --intact_hand Right --config_name modular --camera 0
  python s7_online_learning.py --save_input --person_dir <person_ID> --experiment_name thumbAbAd --intact_hand Right --config_name modular --camera 0
  python s7_online_learning.py --save_input --person_dir <person_ID> --experiment_name indexFlEx --intact_hand Right --config_name modular --camera 0
  python s7_online_learning.py --save_input --person_dir <person_ID> --experiment_name mrpFlEx --intact_hand Right --config_name modular --camera 0
  python s7_online_learning.py --save_input --person_dir <person_ID> --experiment_name fingersFlEx --intact_hand Right --config_name modular --camera 0
  python s7_online_learning.py --save_input --person_dir <person_ID> --experiment_name handOpCl --intact_hand Right --config_name modular --camera 0
  python s7_online_learning.py --save_input --person_dir <person_ID> --experiment_name pinchOpCl --intact_hand Right --config_name modular --camera 0
  python s7_online_learning.py --save_input --person_dir <person_ID> --experiment_name pointOpCl --intact_hand Right --config_name modular --camera 0
  python s7_online_learning.py --save_input --person_dir <person_ID> --experiment_name keyOpCl --intact_hand Right --config_name modular --camera 0
  python s7_online_learning.py --save_input --person_dir <person_ID> --experiment_name indexFlDigitsEx --intact_hand Right --config_name modular --camera 0
  python s7_online_learning.py --save_input --person_dir <person_ID> --experiment_name wristFlEx --intact_hand Right --config_name modular --camera 0
  python s7_online_learning.py --save_input --person_dir <person_ID> --experiment_name wristFlHandCl --intact_hand Right --config_name modular --camera 0
```

## Processing Data
For the first ten recordings, the parameters don't need to be changed, run the processing as follows:
```
  python s3_process_video.py --data_dir data/<person_ID>/recordings/thumbFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process
  python s3_process_video.py --data_dir data/<person_ID>/recordings/thumbAbAd --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process
  python s3_process_video.py --data_dir data/<person_ID>/recordings/indexFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process
  python s3_process_video.py --data_dir data/<person_ID>/recordings/mrpFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process
  python s3_process_video.py --data_dir data/<person_ID>/recordings/fingersFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process
  python s3_process_video.py --data_dir data/<person_ID>/recordings/handOpCl --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process
  python s3_process_video.py --data_dir data/<person_ID>/recordings/pinchOpCl --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process
  python s3_process_video.py --data_dir data/<person_ID>/recordings/pointOpCl --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process
  python s3_process_video.py --data_dir data/<person_ID>/recordings/keyOpCl --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process
  python s3_process_video.py --data_dir data/<person_ID>/recordings/indexFlDigitsEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process
```

For the last two recordings, the parameters need to be changed. Use these commands to check when the actual movement starts:
```
  python s3_process_video.py --data_dir data/<person_ID>/recordings/wristFlEx --experiment_name 1
  python s3_process_video.py --data_dir data/<person_ID>/recordings/wristFlHandCl --experiment_name 1
```

And then run the processing with the found <movement_start> value:
```
 python s3_process_video.py --data_dir data/<person_ID>/recordings/wristFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start <movement_start> --video_end <movement_start + 1800> --process
 python s3_process_video.py --data_dir data/<person_ID>/recordings/wristFlHandCl --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start <movement_start> --video_end <movement_start + 1800> --process
```

## Initial Training
Visualize and check the data:
```
 python s4_train.py --person_dir <person_ID> --intact_hand Right --config_name modular -v
```
Train the model:
```
python s4_train.py --person_dir <person_ID> --intact_hand Right --config_name modular -t -s
```
  
## Online Training
Run online training without perturbation for 6 minutes. 15 seconds per movement following the order above, then repeat in inverted order.
```
python s7_online_learning.py --person_dir <person_ID> --intact_hand Right --config_name modular_online --save_model --experiment_name non_perturb --camera 0 --calibration_frames 60 --allow_tf32
```
Do the same with perturbation:
```
python s7_online_learning.py --person_dir <person_ID> --intact_hand Right --config_name modular_online --save_model --experiment_name perturb --camera 0 --calibration_frames 60 --perturb --allow_tf32
```

## Inference
Run inference for the model after initial training:
```
python s5_inference.py -e --person_dir <person_ID> --config_name modular_online.yaml --model_path /home/haptix/haptix/biomech_PCP/data/<person_ID>/online_trials/non_perturb/models/<person_ID>-online_0.pt
```
Run inference for the model after online training:
```
python s5_inference.py -e --person_dir <person_ID> --config_name modular_online.yaml --model_path /home/haptix/haptix/biomech_PCP/data/<person_ID>/online_trials/non_perturb/models/<person_ID>-online_last.pt
```
To try End-to-End:
```
python s4_train.py --person_dir <person_ID> --intact_hand Right --config_name EtE -t -s
python s5_inference.py -e --person_dir <person_ID> --config_name EtE.yaml --model_path /home/haptix/haptix/biomech_PCP/data/<person_ID>/models/<person_ID>_EtE.pt
```





























