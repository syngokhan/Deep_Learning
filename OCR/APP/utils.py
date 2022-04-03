import os
import settings
import cv2
import numpy as np

def save_upload_image(fileObj):
    filename = fileObj.filename
    name, ext = filename.split(".")
    save_filename = "upload."+ext
    upload_image_path = settings.join_path(settings.SAVE_DIR,
                                           save_filename)

    fileObj.save(upload_image_path)   
    return upload_image_path

def array_to_json_format(arr):
    points = []
    for pt in arr.tolist():
        points.append({"x":pt[0],"y":pt[1]})
    return points

class DocumentScan():
    def __init__(self):
        pass

    @staticmethod
    def resizer(image,width = 500):
        h,w,c = image.shape
        
        height = int((h/w) * width)
        size = (width,height)
        image = cv2.resize(image,(width,height))
        return image ,size

    @staticmethod
    def reorder(point):
        myPointNew = np.zeros_like(point)
        
        myPoints = point.reshape((4,2))
        
        add = myPoints.sum(1)
        
        myPointNew[0] = myPoints[np.argmin(add)]
        myPointNew[3] = myPoints[np.argmax(add)]
        
        diff = np.diff(myPoints,axis = 1)
        
        myPointNew[1] = myPoints[np.argmin(diff)]
        myPointNew[2] = myPoints[np.argmax(diff)]
        
        return myPointNew

    @staticmethod 
    def wrapImg(image,points,w,h):
        
        pts1 = np.float32(points)
        pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        output = cv2.warpPerspective(image,matrix,(w,h))
        return output

    @staticmethod
    def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
        
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def document_scanner(self,image_path):

        self.image = cv2.imread(image_path)
        img_re,self.size = self.resizer(self.image)
        filename ="resize_image.jpg"
        RESIZE_IMAGE_PATH = settings.join_path(settings.MEDIA_DIR,filename)
        
        cv2.imwrite(RESIZE_IMAGE_PATH,img_re)

        try:
            detail = cv2.detailEnhance(img_re,sigma_s = 20, sigma_r = 0.15)
            gray = cv2.cvtColor(detail,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),0)
            
            edge_img = cv2.Canny(blur,75,200)
            
            kernel = np.ones((5,5),np.uint8)
            dilate = cv2.dilate(edge_img,kernel,iterations = 1)
            closing = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
            
            contours,hierarchy = cv2.findContours(closing,
                                                cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
            
            contours = sorted(contours,key = cv2.contourArea,reverse=True)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:
                    peri = cv2.arcLength(cnt,True)
                    approx = cv2.approxPolyDP(cnt,0.02*peri,True)
                    if len(approx) == 4:
                        last_approx = approx
                        break   

            return last_approx,self.size

        except:
            return None,self.size
        
    def calibrate_to_original_size(self,four_points):
        re_four_points = self.reorder(four_points)
        
        multiplier = self.image.shape[1] / self.size[0]
        four_points_orig = re_four_points * multiplier
        four_points_orig = four_points_orig.astype(int)
        
        w,h = self.image.shape[1],self.image.shape[0]
        
        wrap_img = self.wrapImg(self.image,four_points_orig,w,h)

        magic_color = self.apply_brightness_contrast(wrap_img,brightness=40,contrast=40)

        return wrap_img

