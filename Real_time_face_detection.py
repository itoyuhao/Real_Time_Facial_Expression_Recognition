import cv2
import time
import tensorflow
import detect_face
import numpy as np
import random
from tensorflow import keras


# ----tensorflow version check
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
print("Tensorflow version: ",tf.__version__)


def video_init(is_2_write=False,save_path=None):
    writer = None
    cap = cv2.VideoCapture(0)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # default 480
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # default 640

    # width = 480
    # height = 640
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    '''
    ref:https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
    FourCC is a 4-byte code used to specify the video codec. 
    The list of available codes can be found in fourcc.org. 
    It is platform dependent. The following codecs work fine for me.
    In Fedora: DIVX, XVID, MJPG, X264, WMV1, WMV2. (XVID is more preferable. MJPG results in high size video. X264 gives very small size video)
    In Windows: DIVX (More to be tested and added)
    In OSX: MJPG (.mp4), DIVX (.avi), X264 (.mkv).
    FourCC code is passed as `cv.VideoWriter_fourcc('M','J','P','G')or cv.VideoWriter_fourcc(*'MJPG')` for MJPG.
    '''

    if is_2_write is True:
        #fourcc = cv2.VideoWriter_fourcc('x', 'v', 'i', 'd')
        #fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        fourcc = cv2.VideoWriter_fourcc(*'divx')
        if save_path is None:
            save_path = 'demo.avi'
        writer = cv2.VideoWriter(save_path, fourcc, 30, (int(width), int(height)))

    return cap,height,width,writer

def face_detection_MTCNN(detect_multiple_faces=False):
    #----var
    frame_count = 0
    FPS = "Initialing"
    no_face_str = "No faces detected"

    #----video streaming init
    cap, height, width, writer = video_init(is_2_write=False)

    #----MTCNN init
    color = (0,255,0)
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    with tf.Graph().as_default():
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                )

        # 不限制GPU的資源使用
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.1
        sess = tf.Session(config=config)
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    #---CNN recognition

    # Load CNN model (facial expression recognition)
    model = keras.models.load_model('model_optimal_singlechannel_addingdata_downsample_happy0626.h5')

    emotion_dict = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}

    # ---Q decides how many images you want to save simultaneously when you press 's' key
    Q = 3

    graph = detect_face.Graph(100, 60)
    prev_frame = np.zeros((480, 640), np.uint8)

    while (cap.isOpened()):
        #----get image
        ret, img = cap.read()

        if ret is True:
            #----image processing
            # img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # BGR -> RGB
            # print("image shape:",img_rgb.shape)

            #----face detection
            t_1 = time.time()
            bounding_boxes, points = detect_face.detect_face(img_rgb, minsize, pnet, rnet, onet, threshold, factor)
            d_t = time.time() - t_1
            # print("Time of face detection: ",d_t)

            #----bounding boxes processing
            nrof_faces = bounding_boxes.shape[0]
            # print("bounding_boxes:", bounding_boxes)
            if nrof_faces > 0:
                points = np.array(points)
                points = np.transpose(points, [1, 0])
                points = points.astype(np.int16)

                det = bounding_boxes[:, 0:4]
                det_arr = []
                img_size = np.asarray(img.shape)[0:2]
                if nrof_faces > 1:
                    if detect_multiple_faces:
                        for i in range(nrof_faces):
                            det_arr.append(np.squeeze(det[i]))
                    else:
                        bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                        img_center = img_size / 2
                        offsets = np.vstack(
                            [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                        index = np.argmax(
                            bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                        det_arr.append(det[index, :])
                else:
                    det_arr.append(np.squeeze(det))

                #---- Emotion Recognition
                result = []
                confidence = []
                for face in det:
                    x, y, w, h = face
                    crop_img = img_rgb[int(y):int(y + h*3/4), int(x):int(x+w/2)]
                    crop_img = cv2.resize(crop_img, (48, 48))
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

                    # ---- face shot! (press 's' key for saving your face img which is 48 * 48, grayscale)
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        print('-----CHEESE!-----')
                        for _ in range(Q):
                            cv2.imwrite(f'shot_{random.randrange(100000, 999999)}.jpg', crop_img)
                        print(f'{Q} face(s) saved!')

                    # --- Predicting the class of facial expression
                    crop_img = np.reshape(crop_img, [1, 48, 48, 1])
                    classes = model.predict_classes(crop_img / 255)
                    result.append(emotion_dict[classes[0]])
                    scores_arr = np.round(model.predict(crop_img / 255) * 100, decimals=1)
                    scores = round(np.max(model.predict(crop_img / 255))*100, 1)
                    confidence.append(scores)

                    # --- Output the scores
                    happy_score = scores_arr[0,2]
                    print('happy', happy_score)
                    angry_score = scores_arr[0,0]
                    print('angry', angry_score)
                    sad_score = scores_arr[0,4]
                    print('sad', sad_score)
                    neutral_score = scores_arr[0,3]
                    print('neutral', neutral_score)
                    surprise_score = scores_arr[0, 5]
                    print('surprise', surprise_score)

                det_arr = np.array(det_arr)
                det_arr = det_arr.astype(np.int16)

                for i, det in enumerate(det_arr):
                    cv2.rectangle(img, (det[0],det[1]), (det[2],det[3]), color, 1)
                    # ----put the emotion label(s) on the rectangle(s)
                    cv2.putText(img, result[i], (det[0]-2, det[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_8)
                    cv2.putText(img, f'{confidence[i]}%', (det[0]+2, det[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 1, cv2.LINE_8)

            # ----no faces detected
            else:
                cv2.putText(img, no_face_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # ----FPS count
            if frame_count == 0:
                t_start = time.time()
            frame_count += 1
            if frame_count >= 20:
                FPS = "FPS=%1f" % (frame_count / (time.time() - t_start))
                frame_count = 0

            # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
            cv2.putText(img, FPS, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # ----real-time chart
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (25, 25), None)
                diff = cv2.absdiff(prev_frame, gray)
                difference = np.sum(diff)
                prev_frame = gray
                graph.update_frame(int(happy_score), int(angry_score/2+sad_score/2))
                roi = img[-70:-10, -110:-10, :]
                roi[:] = graph.get_graph()
            except:
                pass

            # ----image display
            cv2.imshow("Demo", img)

            # ----image writing
            if writer is not None:
                writer.write(img)

            # ----'q' key pressed?
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("get image failed")
            break

    # ----release
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    face_detection_MTCNN(detect_multiple_faces=True)
    np.load = np_load_old
