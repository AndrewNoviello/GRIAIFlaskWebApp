import io
import tempfile

import flask
import requests
import shutil
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
import googleapiclient.discovery
from google_auth import build_credentials, get_user_info
from tf_detector import TFDetector
from tf_detector import DEFAULT_DETECTOR_LABEL_MAP, DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD
from werkzeug.utils import secure_filename
import tqdm
from PIL import Image, ImageFile, ImageFont, ImageDraw
from yolo_detection.yolo_utils import *

app = flask.Blueprint('google_drive', __name__)


def build_drive_api_v3():
    credentials = build_credentials()
    return googleapiclient.discovery.build('drive', 'v3', credentials=credentials)

def save_image(file_name, mime_type, file_data):
    drive_api = build_drive_api_v3().files()

    generate_ids_result = drive_api.generateIds(count=1).execute()
    file_id = generate_ids_result['ids'][0]

    body = {
        'id': file_id,
        'name': file_name,
        'mimeType': mime_type,
    }

    media_body = MediaIoBaseUpload(file_data,
                                   mimetype=mime_type,
                                   resumable=True)

    drive_api.create(body=body,
                     media_body=media_body,
                     fields='id,name,mimeType,createdTime,modifiedTime').execute()

    return file_id


@app.route('/gdrive/upload', methods=['GET', 'POST'])
def upload_file():
    if 'file' not in flask.request.files:
        return flask.redirect('/dashboard')

    file = flask.request.files['file']
    if (not file):
        return flask.redirect('/dashboard')

    filename = secure_filename(file.filename)

    fp = tempfile.TemporaryFile()
    ch = file.read()
    fp.write(ch)
    fp.seek(0)

    mime_type = flask.request.headers['Content-Type']
    save_image(filename, mime_type, fp)

    return flask.redirect('/')

@app.route('/classify/yolo_test', methods=['GET', 'POST'])
def classify_yolo2():
    print("Calling Classify Endpoint")
    print("YOLO Test: Classifying New Batch of Images")
    service = build_drive_api_v3()
    src = '1-zUhfWJNLAfNx1NRXkkrYm-sr_tY646r' #Test_YOLO_Detect
    trg = '13rwbKEXATmFgLY2pNdGrtNrBskZIeaDb' #Test_YOLO_Output
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='./firstgrimodel_weights.pt')
    query = f"parents = '{src}'"
    response = service.files().list(q=query).execute()
    print("Initial Server Response: ", response)
    files = response.get('files')
    imgs = []
    for f in files: #for path, im, im0s, s in dataset:
        file_id = f['id']
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print("Download %d%%." % int(status.progress() * 100))
        try:
            image = Image.open(fh)
            imgs.append(image)
        except Exception as e:
            print("Image can't be loaded")
            continue
    result = model(imgs)
    print(type(result))
    use_ct = 0
    for pic in result.pandas().xyxy:
        desc = ""
        animalCounts = pic[pic['confidence'] >= .5]['name'].value_counts()
        for a in animalCounts.keys():
            desc += (str(animalCounts[a]) + " " + a + "\n")
        metadata = dict(description=desc)
        print("Current Image MetaData: ", metadata)
        service.files().update(
            fileId=files[use_ct]['id'],
            body=metadata).execute()
        use_ct += 1

    print("Finished classifications; redirecting...")
    return flask.redirect('/dashboard')

##This route is no longer used...
@app.route('/classify/yolo', methods=['GET', 'POST'])
def classify_yolo():
    print("YOLO: Classifying New Batch of Images")
    service = build_drive_api_v3()
    src = '1-zUhfWJNLAfNx1NRXkkrYm-sr_tY646r' #Test_YOLO_Detect
    trg = '13rwbKEXATmFgLY2pNdGrtNrBskZIeaDb' #Test_YOLO_Output

    save_txt = False
    save_conf = False
    classes = None
    update = False

    colors = Colors()
    output_path = ''

    path_to_weights = './best.pt'

    # Load model
    imgsz = (640, 640)
    model = DetectMultiBackend(path_to_weights)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load Images
    bs = 1  # batch_size

    query = f"parents = '{src}'"
    response = service.files().list(q=query).execute()
    print("Initial Server Response: ", response)
    files = response.get('files')
    dt, seen = [0.0, 0.0, 0.0], 0
    ct = 0
    for f in files: #for path, im, im0s, s in dataset:
        file_id = f['id']
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print("Download %d%%." % int(status.progress() * 100))
        try:
            image = Image.open(fh)
            print(image.size)
            image = np.array(image)
            image = np.rollaxis(image, 2, 0)
            print(image.shape)
        except Exception as e:
            print("Image can't be loaded")
            continue
        im = torch.from_numpy(image)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        print(im.shape)
        # Inference
        print(type(model))
        pred = model(im, augment=False)

        # NMS
        #pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)
        print(pred)
        # Process predictions
        print(len(pred))
        '''
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0 = f['name'], im
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'  ## Or 'names[c]' if we want to not show labels in output images
                    annotator.box_label(xyxy, label=label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            # Save results (image with detections)
            cv2.imwrite(output_path + str(ct) + '.jpg', im0)
            ct += 1
        '''
        # Print time (inference-only)
        print(f'Done. ')
    print("Finished classifications; redirecting...")
    return flask.redirect('/dashboard')



@app.route('/classify/megadetector', methods=['GET', 'POST'])
def classify_megadetector():
    service = build_drive_api_v3()
    print("Classifying New Batch of Images")
    src = '17xMEyiGKCJ8m7T0oer1uVaaxohTY2XRq'
    animals_folder = '1f4OUEj8uCRZEH5MDG2J2uFcg3MKTOnGz'
    nonanimals_folder = '1s-qCS5GrPqkQtBh_nuIbSFCYOY0cn5Xm'
    query = f"parents = '{src}'"
    response = service.files().list(q=query).execute()
    print("Initial Server Response: ", response)
    files = response.get('files')
    print("Files to classify: ", files)
    print("Loading Detector")
    detector = TFDetector('./megadetector_model.pb')
    print("Finished Loading Detector")
    detection_results = []
    for f in files:
        file_id = f['id']
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print("Download %d%%." % int(status.progress() * 100))
        try:
            image = Image.open(fh)
        except Exception as e:
            print("Image can't be loaded")
            continue
        try:
            result = detector.generate_detections_one_image(image, f['name'],
                                                            detection_threshold=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD)
            detection_results.append(result)
            count = len(result['detections'])
            print("result: ", result)
            print("count: ", count)
            for i in range(len(result['detections'])):
                print(result['detections'][i]['category'])
                print(type(result['detections'][i]['category']))
                if result['detections'][i]['category'] != '1' and result['detections'][i]['category'] != 1:
                    print("Decreasing count")
                    count = count - 1

            if count > 0:
                print("Picture had animals")
                service.files().update(
                    fileId=f.get('id'),
                    addParents=animals_folder,
                    removeParents=src
                ).execute()
            else:
                print("Picture had no animals")
                service.files().update(
                    fileId=f.get('id'),
                    addParents=nonanimals_folder,
                    removeParents=src
                ).execute()

        except Exception as e:
            print('An error occurred while running the detector on image {}. Exception: {}'.format(f['name'], e))
            continue

    return flask.redirect('/dashboard')

@app.route('/google/moveFile', methods=['GET', 'POST'])
def moveFile():
    service = build_drive_api_v3()
    src = '17xMEyiGKCJ8m7T0oer1uVaaxohTY2XRq'
    trg = '1VlOuYNTQ4fYGKkXpJ4gK4R0cnjtCq8lr'
    query = f"parents = '{src}'"
    response = service.files().list(q=query).execute()
    files = response.get('files')
    nextPageToken = response.get('nextPageToken')

    while nextPageToken:
        response = service.files().list(q=query, pageToken=nextPageToken).execute()
        files.extend(response.get('files'))
        nextPageToken = response.get('nextPageToken')

    for f in files:
        if f['mimeType'] != 'application / vnd.google-apps.folder':
            service.files().update(
                fileId = f.get('id'),
                addParents = trg,
                removeParents = src
            ).execute()

    return flask.redirect('/')

@app.route('/viewIm/<file_id>', methods=['GET'])
def view_file(file_id):
    drive_api = build_drive_api_v3()

    metadata = drive_api.get(fields="name,mimeType", fileId=file_id).execute()

    request = drive_api.get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while done is False:
        status, done = downloader.next_chunk()

    fh.seek(0)

    return flask.send_file(
                     fh,
                     attachment_filename=metadata['name'],
                     mimetype=metadata['mimeType']
               )