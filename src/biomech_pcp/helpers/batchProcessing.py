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
        "python s3_process_video.py --data_dir data/mikey/recordings/thumbFlEx --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/thumbAbAd --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/thumbOpp --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/indexFlEx --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/mrpFlEx --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/midFlEx --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/ringFlEx --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/pinkyFlEx --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/fingersFlEx --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/peaceOpCl --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/fancyOpCl --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/handOpCl --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/pinchOpCl --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/pointOpCl --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/keyOpCl --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/indexFlDigitsEx --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/pinchFlex --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process",
        "python s3_process_video.py --data_dir data/mikey/recordings/pinchAb --experiment_name 1 --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process"
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
