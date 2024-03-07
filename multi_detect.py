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
    # read source streams 
    with open("source.streams","r") as f:
        sources = f.readlines()
    sources = [item.strip("\n") for item in sources]

    # define list to store all threads
    threads  = []

    # loop all source
    for source in sources:
        # command
        try:
            source = int(source)
            command = f"python3 single_detect.py --camera {source}"
        except:
            if source.endswith('.mp4'):
                command = f'python3 single_detect.py --video-file "{source}"'
            elif source.startswith('rtsp'):
                command = f'python3 single_detect.py --rtsp "{source}"'
            elif source.startswith('http://www.youtube.com'):
                command = f'python3 single_detect.py --youtube "{source}"'
            elif source.rstrip() == '':
                print('please prevent empty lines in source.streams')
                continue
            else:
                raise NotImplementedError
        # thread
        thread = threading.Thread(target=run_command, args=(command,))
        threads.append(thread)        

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
