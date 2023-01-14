import torch
import numpy as np
import cv2
from time import time
import argparse
import threading

class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, url, out_file="output.webm"):
  
        self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.backends.cudnn.benchmark = True
            self.model.to(torch.device('cuda'))
        else:
            self.model.to(torch.device('cpu'))
            
        print(f"{self.device} available")

    def get_video_from_url(self):

        return cv2.VideoCapture(self._URL)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        
        frame = [frame]
        results = self.model(frame)
        if self.device  == "cpu":
            labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        else:
            labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_from_url()
        
#         x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
#         y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try:
            fourcc = cv2.VideoWriter_fourcc(*'VP90')
        except Exception as e:
            print(e)
            return
        out = cv2.VideoWriter(self.out_file, fourcc, 20, (1920, 1080))
        while True:
            start_time = time()
            ret, frame = cap.read()
            if not ret:
                break
            results = self.score_frame(frame)
            lab = {}
            for label in results[0]:
                name =  self.class_to_label(label)
                if name not in lab:
                    lab[name]=0
                lab[name]+=1
            # print(lab)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            out.write(frame)
#             print(f"Frames Per Second : {fps}")
#             break
#             cv2.imshow("Object Detection", frame)
#             cv2.waitKey(1)



def parallel_execution(i,args):
    start_time = time()
    a = ObjectDetection(args.input_stream)
    a()
    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Time taken by the function ({i}):{elapsed_time} ")


    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_stream', type=str, help="The destination IP address to use")
    args = parser.parse_args()

    # Create a new object and execute.
    
    overall_start_time = time()
    threads = []
    for i in range(10):
        t = threading.Thread(target=parallel_execution, args=(i,args))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()
        
    overall_end_time = time()
    overall_elapsed_time = overall_end_time - overall_start_time
    print(f"Overall time taken: {overall_elapsed_time}")







if __name__ == '__main__':
    main()
