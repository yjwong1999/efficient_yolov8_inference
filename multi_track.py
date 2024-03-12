import threading
import subprocess
import argparse

def run_command(command):
    """
    Execute a command in the shell.
    """
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

def main(opt):
    # read source.streams 
    with open("source.streams","r") as f:
        sources = f.readlines()
    sources = [item.strip("\n") for item in sources]

    # read geofencing.streams 
    if opt.geofencing:
        with open("geofencing.streams","r") as f:
            geofencings = f.readlines()
        geofencings = [item.strip("\n") for item in geofencings]    
    
        # make sure the geofencing ROI are correct
        assert len(sources) == len(geofencings), 'Please provide the corresponding geofencing ROI for each video streaming source'
    else:
        geofencings = [None] * len(sources)

    # try if drive utility function is working
    # before multithreading
    if opt.geofencing:
        try:
            from counter import Counter
            counter = Counter(0,0,0,0,0)    # init a dummy counter
            counter.drive_handler.post()    # try posting smtg to see if API working
            del counter                     # delete the dummy counter
        except Exception as e:
            print(e)
            import time
            time.sleep(3)

    # define list to store all threads
    threads  = []

    # loop all source
    for stream_idx, (source, geofencing) in enumerate(zip(sources, geofencings)):
        # command
        try:
            source = int(source)
            command = f'python3 single_track.py --camera {source} --stream-idx {stream_idx}'
        except:
            if source.endswith('.mp4'):
                command = f'python3 single_track.py --video-file "{source}" --stream-idx {stream_idx}'
            elif source.startswith('rtsp'):
                command = f'python3 single_track.py --rtsp "{source}" --stream-idx {stream_idx}'
            elif source.startswith('http://www.youtube.com'):
                command = f'python3 single_track.py --youtube "{source}" --stream-idx {stream_idx}'
            elif source.rstrip() == '':
                print('please prevent empty lines in source.streams')
                continue
            else:
                raise NotImplementedError

        # if geofencing
        if opt.geofencing:
            command += f' --roi-xyxy "{geofencing}"'
        
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--geofencing', action='store_true',
                        help='if flagged, activate geofencing')
    opt = parser.parse_args()

    main(opt)
