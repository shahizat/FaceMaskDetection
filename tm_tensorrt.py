import cv2
import tensorrt as trt
import numpy as np
import common
import platform as plt
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

def preprocess(img):
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img_resized = cv2.resize(img, (224,224))
    data[0] = (img_resized.astype(np.float32) / 127.0) - 1

    return data

def load_engine(trt_runtime, engine_path):
    
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)

    return engine

def get_label(label_path):

    label = {}
    with open(label_path) as f:
        for line in f.readlines():
            idx,name = line.strip().split(' ')
            label[int(idx)] = name
    
    return label

def main(win_title):

    # load trt engine
    print('load trt engine')
    trt_path = 'engine.trt'
    engine = load_engine(trt_runtime, trt_path)

    print('load labels')
    label = get_label('keras_models/labels.txt')

    # allocate buffers
    print('allocate buffers')
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    print('create execution context')
    context = engine.create_execution_context()

    print('start stream')
    fps = -1
    GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'

    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    while(1):

        t_start = time.time()
        ret, frame = cap.read()
        size = (224, 224)
        inputs[0].host = preprocess(frame)

        # with engine.create_execution_context() as context:
        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        
        preds = trt_outputs[0]

        idx = np.argmax(preds)

        result = label[idx]

        info = '{} : {:.3f} , FPS {}'.format(result, preds[idx], fps)

        cv2.putText(frame, info, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4)

        cv2.imshow(win_title, frame)

        if cv2.waitKey(1) == ord('q'):
            break

        fps = int(1/(time.time()-t_start))
    
    cap.release()
    cv2.destroyAllWindows()
    print('Quit')

if __name__ == '__main__':
    
    main(plt.system()+' - TensorRT')
