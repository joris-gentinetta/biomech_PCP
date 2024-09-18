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
        "python s3_process_video.py --data_dir data/P7_453/recordings/thumbFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/thumbAbAd --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/indexFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/mrpFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/fingersFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/handOpCl --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/pinchOpCl --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/pointOpCl --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/keyOpCl --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/indexFlDigitsEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/freeMovement1 --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/freeMovement2 --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/freeMovement3 --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 10 --video_end 1810 --process",
        # "python s3_process_video.py --data_dir data/P7_453/recordings/wristFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 230 --video_end 2030 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/wristFlEx --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 750 --video_end 2550 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/wristFlHandCl --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 235 --video_end 2035 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/wristFlHandOp --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 340 --video_end 2140 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/wristFlEx_up --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 750 --video_end 2550 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/wristFlHandCl_up --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 235 --video_end 2035 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/wristFlHandOp_up --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 340 --video_end 2140 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/boxBlocksFree --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 175 --video_end 1985 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/boxBlocksActual --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 240 --video_end 2040 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/mugGrasp --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 270 --video_end 2070 --process",
        "python s3_process_video.py --data_dir data/P7_453/recordings/scissorGrip --experiment_name 1  --intact_hand Right --plane_frames_start 10 --plane_frames_end 110 --video_start 270 --video_end 2070 --process",

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
