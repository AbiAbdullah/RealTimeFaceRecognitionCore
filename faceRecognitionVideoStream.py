import face_recognition
import os
import cv2
import time


KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
count = 0
MODEL = 'cnn'  # 'hog' or 'cnn' - CUDA accelerated (if available) deep-learning pretrained model
outputPath = 'output'


# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;http"

streamUrl= "" # add rtsp url 
video = cv2.VideoCapture(streamUrl, cv2.CAP_FFMPEG)
# video = cv2.VideoCapture(0)


# Return (R,G,B) from color from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color


print('Loading known faces...')
known_faces = []
known_names = []

# We oranize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)
# print('Processing unknown faces...')
# Now let's loop over a folder of faces we want to label
# for filename in os.listdir(UNKNOWN_FACES_DIR):
while True:
    # Load image
    # print(f'Filename {filename}', end='')
    # image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')
    try:
        video.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
        ret, image = video.read()
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
           # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        count+=1

    except Exception as e:
        print(str(e))
    # ret, image = video.read()
    if ret == False:
        print('Frame is Empty')
        break;

  
    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)

    # Now since we know loctions, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    encodings = face_recognition.face_encodings(rgb_small_frame, locations)

    # We passed our image through face_locations and face_encodings, so we can modify it
    # First we need to convert it from RGB to BGR as we are going to work with cv2
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)        

# But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
    # print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):

        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        # Since order is being preserved, we check if any face was found then grab index
        # then label (name) of first matching known face withing a tolerance
        match = None
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            # print(f' - {match} from {results}')
        else:
            match='Unknown'
            # cv2.imwrite(match+str(time.time())+'.jpg',image)
            # cv2.imwrite(os.path.join(outputPath, match+str(time.time())+'.jpg' ), image)
            


        # Each location contains positions in order: top, right, bottom, left
        # top_left = (face_location[3], face_location[0])
        # bottom_right = (face_location[1], face_location[2])
        top_left = (face_location[3] * 4, face_location[0] * 4)
        bottom_right = (face_location[1] * 4, (face_location[2] * 4)+22)

        # Get color by name using our fancy function
        color = name_to_color(match)

        # Paint frame
        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

        # Now we need smaller, filled grame below for a name
        # This time we use bottom in both corners - to start from bottom and move 50 pixels down
        top_left = (face_location[3] * 4, face_location[2] *4)
        bottom_right = (face_location[1] * 4, (face_location[2] * 4)+33)

        # Paint frame
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

        # Wite a name
        cv2.putText(image, match, (face_location[3] * 4 + 10, face_location[2]* 4 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    # Show image
    cv2.imshow(filename, image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
    # cv2.waitKey(0)
    # cv2.destroyWindow(filename)