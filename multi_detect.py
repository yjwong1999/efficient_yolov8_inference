import threading
import subprocess

def run_command(command):
    """
    Execute a command in the shell.
    """
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

def main():
    # Define your command (replace 'abc.py' with the actual script name)
    command1 = "python3 single_detect.py --video-file dataset_cam1.mp4"
    command2 = "python3 single_detect.py --webcam"

    # Create two threads
    thread1 = threading.Thread(target=run_command, args=(command1,))
    thread2 = threading.Thread(target=run_command, args=(command2,))

    # Start both threads
    thread1.start()
    thread2.start()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()

if __name__ == "__main__":
    with open("source.streams","r") as f:
        lines = f.readlines()
    print(lines)

    main()
