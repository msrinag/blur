import torch
from PIL import Image
import cv2
from tqdm import tqdm

CKPT_PATH = r'C:/Users/srii ideapad/Downloads/yolov5-master/yolov5-master/best.pt'


yolov5 = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
yolov5.eval()

model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
classes = ['person',]

model.eval()


# Define video writer

print('confidence: ' + str(yolov5.conf))

cap = cv2.VideoCapture(r'C:/Users/srii ideapad/Downloads/traficcam.mp4')
# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
j=1
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in tqdm(range(num_frames)):
    # Read frame from video
    ret, frame = cap.read()
    #print(j)
    j+=1
    if not ret:
        break
    #print(frame.shape)
    results = model(frame, size=640)
    detections = results.pandas().xyxy[0]

    # Filter the detections to only include 'person' and 'license plate'
    filtered_detections = detections[detections['name'].isin(classes)]

    results = yolov5(frame)
    # Draw bounding boxes around the detected objects
    for _, detection in filtered_detections.iterrows():
        x_min, y_min, x_max, y_max = detection[['xmin', 'ymin', 'xmax', 'ymax']]
        class_name = detection['name']
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    
    #print(results.xyxy[0])
    for i, row in results.pandas().xyxy[0].iterrows():
        if row['confidence'] < yolov5.conf:
            print("low")
            pass

        cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 2)
        cv2.putText(frame, "plate"+str(row['confidence']), (int(row['xmin']), int(row['ymin']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        #cv2.imwrite("test"+str(j)+".jpg",frame)
    
    

    

    # Show frame
    #cv2.imshow('License Plate Detection', frame)
    out.write(frame)
   
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()