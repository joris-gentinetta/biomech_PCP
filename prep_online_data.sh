#!/bin/bash

# Array of person IDs
person_ids=("P_577" "P_407" "P_711" "P_238" "P_426" "P_950" "P_149" "P_668")

# Iterate over each person
for person_id in "${person_ids[@]}"; do
    echo "Processing for $person_id"

    # Run all commands for a person in parallel
    python s3_process_video.py --data_dir data/$person_id/recordings/thumbFlEx --experiment_name 2  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 1810 --video_end 3610 --process &
    python s3_process_video.py --data_dir data/$person_id/recordings/thumbAbAd --experiment_name 2  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 1810 --video_end 3610 --process &
    python s3_process_video.py --data_dir data/$person_id/recordings/indexFlEx --experiment_name 2  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 1810 --video_end 3610 --process &
    python s3_process_video.py --data_dir data/$person_id/recordings/mrpFlEx --experiment_name 2  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 1810 --video_end 3610 --process &
    python s3_process_video.py --data_dir data/$person_id/recordings/fingersFlEx --experiment_name 2  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 1810 --video_end 3610 --process &
    python s3_process_video.py --data_dir data/$person_id/recordings/handOpCl --experiment_name 2  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 1810 --video_end 3610 --process &
    python s3_process_video.py --data_dir data/$person_id/recordings/pinchOpCl --experiment_name 2  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 1810 --video_end 3610 --process &
    python s3_process_video.py --data_dir data/$person_id/recordings/pointOpCl --experiment_name 2  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 1810 --video_end 3610 --process &
    python s3_process_video.py --data_dir data/$person_id/recordings/keyOpCl --experiment_name 2  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 1810 --video_end 3610 --process &
    python s3_process_video.py --data_dir data/$person_id/recordings/indexFlDigitsEx --experiment_name 2  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 1810 --video_end 3610 --process &
    python s3_process_video.py --data_dir data/$person_id/recordings/wristFlEx --experiment_name 2  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 2300 --video_end 4100 --process &
    python s3_process_video.py --data_dir data/$person_id/recordings/wristFlHandCl --experiment_name 2  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 2300 --video_end 4100 --process &

    # Wait for all processes of this person to complete before moving to the next person
    wait
    echo "Finished processing for $person_id"
done

echo "All processing completed."
