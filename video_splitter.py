import cv2 as cv
import os
import sys

def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Please enter the name of the video file you want to split")
        return
    
    args = args[0].split('.')
    vid_split(args[0], args[1])
    print("Done")
    return
    
 
def vid_split(name = 'test', extension = 'mp4'):
    if not os.path.exists(f"./inputs/{name}"):
        os.mkdir(f"./inputs/{name}")

    vid_link = f"./inputs/{name}.{extension}"
    cap = cv.VideoCapture(vid_link)
    frame_no = 0


    while True:
        success, frame = cap.read()
        if not success:
            break

        cv.imwrite(f"./inputs/{name}/{name}_{frame_no}.jpg", frame)
        frame_no += 1

if __name__ == "__main__":
    main()