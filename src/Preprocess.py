import skimage
from skimage.io import imread
import numpy as np
from mtcnn.mtcnn import MTCNN

def preprocess_all(im):
    ycbcr = skimage.color.rgb2ycbcr(im)
    ycbcr = ycbcr/255
    y = ycbcr[:,:,0]
    value = np.sum(y<0.5) / np.sum(y>0.5)
    if value > 5 or value == "inf":
        y = skimage.exposure.adjust_gamma(y, 0.7)
    elif value < 0.2:
        y = skimage.exposure.adjust_gamma(y, 1.3)
    ycbcr[:,:,0] = y
    final = skimage.color.ycbcr2rgb(ycbcr*255)
    final = skimage.filters.unsharp_mask(final, radius=2, amount=1)
    final = np.clip(final, 0,1)*255
    return final

def process_crop(im):
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    ycbcr = skimage.color.rgb2ycbcr(im)
    ycbcr = ycbcr/255
    y = ycbcr[:,:,0]
    sharp= False
    value = np.sum(y<0.5) / np.sum(y>0.5)
    if value > 5 or value == "inf" or value < 0.2:
        sharp = True
    while True:
        value = np.sum(y<0.5) / np.sum(y>0.5)
        if value > 5 or value == "inf":
            y = skimage.exposure.adjust_gamma(y, 0.8)
        elif value < 0.2:
            print(value)
            y = skimage.exposure.adjust_gamma(y, 1.2)
        else:
            break
    ycbcr[:,:,0] = y
    final = skimage.color.ycbcr2rgb(ycbcr*255)
    final = np.clip(final, 0,1)
    # enhanced image = original + amount * (original - blurred)
    if sharp:
        final = skimage.filters.unsharp_mask(final, radius=2, amount=1)
    else:
        final = skimage.filters.unsharp_mask(final, radius=1, amount=0.5)

    return final

def try_rotate(img, detector):
    faces = []
    im = []
    for i in range(0, 360, 90):
        im = skimage.transform.rotate(img, i, resize=True)*255
        im = np.clip(im, 0,255)
        faces = detector.detect_faces(im)
        if len(faces)>0:
            break
    return faces, im, i

def find_face_and_preprocess(path, detector):
    final_imgs = []
    im = imread(path)
    crops = []
    faces = detector.detect_faces(im)
    if len(faces)==0:
        im = preprocess_all(im)
        faces = detector.detect_faces(im)
        if len(faces)==0:
            faces, im, _ = try_rotate(im, detector)
            if len(faces) == 0:
                return None
    for face in faces:
        [X,Y,W,H] = face['box']
        crop = im[Y:Y+H, X:X+W]
        face = detector.detect_faces(crop)
        if len(face) == 0:
            face, crop, rotation = try_rotate(crop, detector)
            if len(face) == 1:
                im = skimage.transform.rotate(im, rotation, resize=True)
                face = detector.detect_faces(im*255)
                [X,Y,W,H] = face[0]['box']
        if len(face) == 1:
            base = len(im[:,1,1])
            h = len(im[1,:,1])
            bord_x = 30
            bord_y = 70
            crop = im[Y:Y+H, X:X+W]
            if (Y-bord_y > 0) and (Y+H+bord_y < h) and (X-bord_x > 0) and (X+W+bord_x < base):
                crops += [im[Y-bord_y:Y+H+bord_y, X-bord_x:X+W+bord_x]]
            else:
                crops += [im[Y:Y+H, X:X+W]]
    if len(crops) == 0:
        print("no faces")
        return None
    else:
        for crop in crops:
            final_crop = process_crop(crop)
            final_imgs += [final_crop]
    return final_imgs