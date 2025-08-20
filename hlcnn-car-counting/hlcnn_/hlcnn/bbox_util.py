import cv2 
import numpy as np

#Drawing bounding boxes on images
#Input is taken in the form of image + bounding box coords
#Output is images with boxes drawn
def draw_rect(im, cords, color = None):

    im = im.copy()
    cords = cords[:,:4]
    cords = cords.reshape(-1,4)

    if not color:
        color = [255,255,255]

    for cord in cords:
        pt1, pt2 = (cord[0], cord[1]) , (cord[2], cord[3])
        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])
        im = cv2.rectangle(im.copy(), pt1, pt2, color, int(max(im.shape[:2])/200))
    return im

#Bounding box area calculation
def bbox_area(bbox):
    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])

#Clipping of bounding boxes within image
#Make sure boxes stay inside image boundaries
#Removes boxes if remaining area visible is less than alpha      
def clip_box(bbox, clip_box, alpha):

    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
    y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
    x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
    y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)
    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))
    delta_area = ((ar_ - bbox_area(bbox))/ar_)
    mask = (delta_area < (1 - alpha)).astype(int)
    bbox = bbox[mask == 1,:]
    return bbox

#Image Rotation around its center
#Expands canvas to avoid cropping
def rotate_im(image, angle):

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    image = cv2.warpAffine(image, M, (nW, nH))
    return image

#Get bounding box corners by conversion of [x1,y1,x2,y2]
def get_corners(bboxes):

    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)
    
    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    
    return corners

#Rotating bounding box corners around image center to match rotate image
def rotate_box(corners,angle,  cx, cy, h, w):
    
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated

#Axis aligned box that fully encloses rotated corners
def get_enclosing_box(corners):
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final

#Resizing image to size without change of aspect ration
def letterbox_image(img, inp_dim):

    inp_dim = (inp_dim, inp_dim)
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h))
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 0)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    return canvas