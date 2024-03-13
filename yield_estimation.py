from pathlib import Path
import cv2
from ultralytics import YOLO
from PIL import Image


cluster_detection_model = YOLO('cluster_detection/v8n2/weights/best.pt')
grapes_detection_model = YOLO('grapes_detection/v8n2/weights/best.pt')
temp_imgs_dir = Path('temp_images')
temp_imgs_dir.mkdir(exist_ok=True)

def yield_estimation(video_path):

    cap = cv2.VideoCapture(video_path)
    clusters_ids = set()
    
    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            results = cluster_detection_model.track(frame, persist=True)
            ids = results[0].boxes.id.cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clusters_ids.update(ids)
    
            new_boxes_indexes = [ids.index(x) for x in ids if x not in clusters_ids]
            new_boxes = [boxes[i] for i in new_boxes_indexes]
    
            for i, box in enumerate(new_boxes):
            
                x1, x2, y1, y2 = int(box[0]), int(box[2]), int(box[1]), int(box[3])
                cropped_image = img[y1:y2, x1:x2]
        
                _cropped_image = Image.fromarray(cropped_image, 'RGB')
                _cropped_image.save(temp_imgs_dir/f'{uuid.uuid4().hex}.jpg')
        else:
            break
    cap.release()

    num_grapes = 0

    for img in temp_imgs_dir:
        results = grapes_detection_model.predict(img)
        boxes = results[0].boxes.xyxy.cpu().tolist()
    
        num_grapes += len(boxes)

    return len(clusters_ids), num_grapes