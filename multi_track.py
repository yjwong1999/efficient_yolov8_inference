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
    # read source.streams 
    with open("source.streams","r") as f:
        sources = f.readlines()
    sources = [item.strip("\n") for item in sources]

    # read geofencing.streams 
    with open("geofencing.streams","r") as f:
        geofencings = f.readlines()
    geofencings = [item.strip("\n") for item in geofencings]    

    # assertion
    if len(geofencings) != 0:
        # make sure the geofencing ROI are correct
        assert len(sources) == len(geofencings), 'Please provide the corresponding geofencing ROI for each video streaming source'

        # try if drive utility function is working
        # before multithreading
        try:
            from counter import Counter
            counter = Counter(0,0,0,0,0)   # init a dummy counter
            counter.drive_handler.post()   # try posting smtg to see if API working
            del counter                    # delete the dummy counter
        except Exception as e:
            print(e)
            input('Please type anything to continue')

    # define list to store all threads
    threads  = []

    # loop all source
    for stream_idx, (source, geofencing) in enumerate(zip(sources, geofencings)):
        # command
        try:
            source = int(source)
            command = f'python3 single_track.py --camera {source} --stream-idx {stream_idx} --roi-xyxy "{geofencing}"'
        except:
            if source.endswith('.mp4'):
                command = f'python3 single_track.py --video-file "{source}" --stream-idx {stream_idx} --roi-xyxy "{geofencing}"'
            elif source.startswith('rtsp'):
                command = f'python3 single_track.py --rtsp "{source}" --stream-idx {stream_idx} --roi-xyxy "{geofencing}"'
            elif source.startswith('http://www.youtube.com'):
                command = f'python3 single_track.py --youtube "{source}" --stream-idx {stream_idx} --roi-xyxy "{geofencing}"'
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

