import multiprocessing
import subprocess

def run_command(command):
    """
    This function runs a command using subprocess.
    """
    process = subprocess.Popen(command, shell=True)
    process.communicate()  # Wait for the process to finish

def main():
    # List of commands to run with different arguments
    commands = [
        "python s3_process_video.py --data_dir data/P5_869/recordings/thumbFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/P5_869/recordings/thumbAbAd --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/P5_869/recordings/thumbOpp --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/P5_869/recordings/indexFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/P5_869/recordings/mrpFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/P5_869/recordings/fingersFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/P5_869/recordings/handOpCl --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/P5_869/recordings/pinchOpCl --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/P5_869/recordings/pointOpCl --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/P5_869/recordings/keyOpCl --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/P5_869/recordings/indexFlDigitsEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/P5_869/recordings/wristFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 240 --video_end 3840 --process",
        "python s3_process_video.py --data_dir data/P5_869/recordings/wristFlHandCl --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 200 --video_end 3800 --process"
]

    # Create a pool of processes
    processes = []
    for command in commands:
        p = multiprocessing.Process(target=run_command, args=(command,))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
