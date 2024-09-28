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
        "python s3_process_video.py --data_dir data/P6_820/recordings/midFlEx --experiment_name 1  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process --jorisThumb",
        "python s3_process_video.py --data_dir data/P6_820/recordings/ringFlEx --experiment_name 1  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process --jorisThumb",
        "python s3_process_video.py --data_dir data/P6_820/recordings/pinkyFlEx --experiment_name 1  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process --jorisThumb",
        "python s3_process_video.py --data_dir data/P6_820/recordings/fingerWave --experiment_name 1  --intact_hand Left --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 3610 --process --jorisThumb"
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
