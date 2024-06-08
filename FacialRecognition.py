# importing libraries

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
#from torchvision import datasets
#from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os
# initializing MTCNN and InceptionResnetV1

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # keep_all=False
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Calculate distance based on pixel size of the bounding box
def calculate_distance(box_width_pixels, known_width_cm, focal_length_pixels):
    return (known_width_cm * focal_length_pixels) / box_width_pixels

# Assuming a known average face width in centimeters and a focal length
known_face_width_cm = 20  # Example: Average face width in centimeters
focal_length = 1000  # Example: Focal length of the camera in pixels

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("fail to grab frame, try again")
        break

    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True)

    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)

        for i, prob in enumerate(prob_list):
            if prob > 0.90:
                box = boxes[i]

                # Calculate the width of the bounding box
                box_width_pixels = abs(box[2] - box[0])

                # Calculate the estimated distance
                estimated_distance = calculate_distance(box_width_pixels, known_face_width_cm, focal_length)
                print(f"Estimated distance: {estimated_distance} centimeters")


# loading data.pt file
load_data = torch.load('data.pt')
embedding_list = load_data[0]
name_list = load_data[1]

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("fail to grab frame, try again")
        break

    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True)

    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)

        for i, prob in enumerate(prob_list):
            if prob > 0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                dist_list = []  # list of matched distances, minimum distance is used to identify the person

                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list)  # get minumum dist value
                min_dist_idx = dist_list.index(min_dist)  # get minumum dist index
                name = name_list[min_dist_idx]  # get name corrosponding to minimum dist

                box = boxes[i]

                original_frame = frame.copy()  # storing copy of frame before drawing on it

                if min_dist < 0.90:
                    #frame = cv2.putText(frame, name + ' ' + str(min_dist), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX,
                                       # 1, (0, 255, 0), 1, cv2.LINE_AA)

                    frame = cv2.putText(frame, name + ' ' + str(min_dist), (int(box[0]), int(box[1])), cv2.LINE_AA,
                                        0.8, (0, 255, 0), 2, cv2.LINE_AA)


                frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)


    cv2.imshow("IMG", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC
        print('Esc pressed, closing...')
        break

    elif k % 256 == 32:  # space to save image
        print('Enter your name :')
        name = input()

        # create directory if not exists
        if not os.path.exists('photos/' + name):
            os.mkdir('photos/' + name)

        img_name = "photos/{}/{}.jpg".format(name, int(time.time()))
        cv2.imwrite(img_name, original_frame)
        print(" saved: {}".format(img_name))

cam.release()
cv2.destroyAllWindows()
