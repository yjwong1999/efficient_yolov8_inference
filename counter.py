##############################
# For Geofencing + Counter 
##############################
import os
import datetime
import pandas as pd

from drive_utils.wrapper import DriveHandler

class Counter:
    def __init__(self, x1, y1, x2, y2, idx):
        """
        Initialize a counter

        Args:
            roi = (x1, x2, y1, y2) which have been normalized to [0,1] range
            x1, y1 ---------------
            |                    |
            |         ROI        |
            |                    |
            --------------- x2, y2

            idx = which camera
        """
        
        self.roi_x1 = x1
        self.roi_y1 = y1
        self.roi_x2 = x2
        self.roi_y2 = y2
        self.idx = idx
        
        self.steps = 0

        self.drive_handler = DriveHandler()

        self.reset()

    def reset(self):
        '''
        Definition of IN and OUT
        
        IN -> the person has entered the room
        OUT -> the person is in the geofencing zone, before entering the room
        
        buffer_in -> to store person who first appeared in the room
        buffer_out -> to store person who first appeared in the geofencing zone, before entering
        '''
        self.move_in = {}
        self.move_out = {} # not implemented yet
        self.count_in = 0
        self.count_out = 0 # not implemented yet
        
        self.buffer_out = {} # to store id of ppl that haven enter yet     
        self.buffer_in = {}  # to store id of ppl that first appear inside

        self.current_date = datetime.datetime.now().date()
        self.current_hour = datetime.datetime.now().hour

        # create new logfile at local
        self.logfile = f'log/camera{str(self.idx).zfill(3)}_{self.current_date.strftime("%Y-%m-%d")}_count.txt'
        if not os.path.isdir('log'):
            os.mkdir('log')

        # if this is a new logfile, means we are on the next day
        if not os.path.isfile(self.logfile):
            # update data until yesterday to google drive
            try:
                self.drive_handler.post()
                pass
            except Exception as e:
                print(e)
                
            # create new logfile for today
            with open(self.logfile, 'w') as f:
                # this will create an empty logile
                pass  
        # if this file existed, means our code has been interrupted
        # now we are restarting the code and resume today counting
        else:
            # fill in the missing values between the interrupted duration
            # and read back the last count   
            try:
                self.resume()
            except:
                pass

    def resume(self):
        # fill in the missing values between the interrupted duration
        # and read back the last count        
        df = pd.read_csv(self.logfile,delimiter = ' ',header = None, engine = 'python')
        df.columns = ['Date','Time','Count']
        df_hr = pd.to_datetime(df['Time'].to_list(),format='%H:%M:%S.%f').hour  
        num_rows , _ = df.shape
    
        # print(f"the number of rows is {num_rows}")
        # print(df_hr[-1])
        # print(self.current_hour)
    
        if num_rows < self.current_hour:
            print("filling missing values")
            with open (self.logfile,'a') as f:
                missHr = self.current_hour - df_hr[-1]
                lastCount = int(df.iloc[-1]['Count'])
                                        
                for i in range(missHr):
                    missTime = datetime.time(df_hr[num_rows-1]+i+1,0,0,1)
                    datetimeInfo = datetime.datetime.combine(datetime.datetime.today().date(),missTime)
                    f.write(f'{datetimeInfo} {lastCount}\n')
                    print("written")
                    
        # update self.count_in = lastCount (before system interrupted and restart)
        lastCount = int(df.iloc[-1]['Count'])
        self.count_in = lastCount            
                                
    def clear_buffer(self):
        # increment steps
        self.steps += 1

        # every 900 steps (assuming 30fps, then is every 30s)
        now = datetime.datetime.now()
        if self.steps % 300:
            # remove IDs in buffer_in 
            remove_ids = []
            for id, timestamp in self.buffer_in.items():
                delta_time = now - timestamp
                
                # remove the id in buffer_id, if the last appearance is more than 30s
                if delta_time.total_seconds() > 30:
                    remove_ids.append(id)
            
            for id in remove_ids:
                del self.buffer_in[id]
                
            # remove IDs in buffer_out
            remove_ids = []
            for id, timestamp in self.buffer_out.items():
                delta_time = now - timestamp
                
                # remove the id in buffer_id, if the last appearance is more than 30s
                if delta_time.total_seconds() > 30:
                    remove_ids.append(id)
            
            for id in remove_ids:
                del self.buffer_out[id]                
        
    def update(self, img_shape=None, pred_boxes=None):
        """
        Update the total number of objects move in/out the ROI

        Args:
            img_shape: the img shape
            pred_boxes: the bbox of predicted obj
        """

        # Update Detect results
        try:
            all_cls  = reversed(pred_boxes.boxes.cls)
            all_conf = reversed(pred_boxes.boxes.conf)
            all_id   = reversed(pred_boxes.boxes.id)
            all_xyxy = reversed(pred_boxes.boxes.xyxy)
        except:
            return None
        
        for cls, conf, id, xyxy in zip(all_cls, all_conf, all_id, all_xyxy):
            c, conf, id = int(cls), float(conf), None if id is None else int(id)
            x1, y1, x2, y2 = xyxy

            # centroid
            x_mid = (x1 + x2) / 2
            y_mid = (y1 + y2) / 2
            
            # conditions
            condition_1 = x_mid >= self.roi_x1 * img_shape[1]
            condition_2 = x_mid <= self.roi_x2 * img_shape[1]
            condition_3 = y_mid >= self.roi_y1 * img_shape[0]
            condition_4 = y_mid <= self.roi_y2 * img_shape[0]
            within_roi = condition_1 and condition_2 and condition_3 and condition_4
            
            # update count
            if id is None:
                return None # tracker will assign None as ID, if the track is not alive long enough
                
            if (within_roi) and (id not in self.buffer_in.keys()):
                self.buffer_out[id] = datetime.datetime.now()
            elif (within_roi) and (id in self.buffer_in.keys()):
                pass
            elif (not within_roi) and (id not in self.buffer_out.keys()):
                self.buffer_in[id] = datetime.datetime.now()
            elif (not within_roi) and (id in self.buffer_out.keys()):
                self.count_in += 1
                self.move_in[id] = datetime.datetime.now()
                del self.buffer_out[id]
        
        # clear buffer
        # self.clear_buffer()

    def log(self):
        # Get the current date and time
        now = datetime.datetime.now()

        # Check if the current time is at the start of a new hour
        if now.hour != self.current_hour:
            # update current hour
            print(now.strftime("%Y-%m-%d %H:%M:%S"))
            self.current_hour = now.hour

            # log total count
            with open(self.logfile, 'a') as f:
                f.write(f'{datetime.datetime.now()} {self.count_in}\n')

        # Check if a new day has passed
        if now.date() > self.current_date:
            # reset
            self.current_date = now.date()
            self.reset()

