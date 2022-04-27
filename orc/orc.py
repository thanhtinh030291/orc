import cv2
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from var_dump import var_dump
from django.core.files.storage import FileSystemStorage, default_storage
import numpy as np
import os
import pytesseract
from PIL import Image
import json
import re
import uuid





from django.conf import settings


# Create your views here.
@csrf_exempt
@api_view(['GET', 'POST'])
def index(request):
    # lấy và lưu file từ client
    file = request.FILES['upload']
    file_name = default_storage.save(file.name, file)

    file = default_storage.open(file_name)
    file_url = default_storage.url(file_name)
    file = file_url[1:]
    img = cv2.imread(file)

    # điều chỉnh kích thước
    image_re = image_resize(img, width = 3000)
    height, width, channels = image_re.shape
    min_h1,max_h1= min_max_h(image_re)

    img = image_re[min_h1-20:max_h1+20, 0:width]


    # ảnh nhị phân (binary image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, img_bin = cv2.threshold(
        gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255 - img_bin

    # xác định các cạnh hàng ngang bằng một hàm đơn giản
    kernel_len = gray.shape[1] // 120
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    image_horizontal = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_horizontal, hor_kernel, iterations=3)

    h_lines = cv2.HoughLinesP( horizontal_lines, 1, np.pi / 180, 30, maxLineGap=250 )

    # nhóm những cạnh này lại thành các cạnh chính

    new_horizontal_lines = group_h_lines(h_lines, kernel_len)



    # # xác định các cạnh hàng dọc bằng một hàm đơn giản
    kernel_len = gray.shape[1] // 120
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    image_vertical = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_vertical, ver_kernel, iterations=3)
    v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi / 180, 30, maxLineGap=250)

    # nhóm những cạnh này lại thành các cạnh chính
    new_vertical_lines = group_v_lines(v_lines, kernel_len)

    # tìm giao điểm
    points = []
    for hline in new_horizontal_lines:
        x1A, y1A, x2A, y2A = hline
        for vline in new_vertical_lines:
            x1B, y1B, x2B, y2B = vline

            line1 = [np.array([x1A, y1A]), np.array([x2A, y2A])]
            line2 = [np.array([x1B, y1B]), np.array([x2B, y2B])]

            x, y = seg_intersect(line1, line2)
            if x1A <= x <= x2A and y1B <= y <= y2B:
                points.append([int(x), int(y)])

    # xác định góc
    cells = []
    i =0
    for point in points:
        i = i+1
        left, top = point
        right_points = sorted( [p for p in points if p[0] > left and p[1] == top], key=lambda x: x[0])
        bottom_points = sorted([p for p in points if p[1] > top and p[0] == left], key=lambda x: x[1])
        right, bottom = get_bottom_right(right_points, bottom_points, points)
        if right and bottom:
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            cells.append([left, top, right, bottom])
    file_name_crop = "media/{}.jpg".format( str(uuid.uuid4()) )
    cv2.imwrite(file_name_crop, img)
    image = cv2.imread(file_name_crop)
    gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gr = cv2.threshold(gr, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    to_text = []
    to_value = []
    cotx0_firt = None
    cotx0_end = None
    text_amt = "Số tiền"
    text_content = "Nội dung"
    for cell in cells:
        if(cotx0_firt == None or cotx0_end == None or cotx0_firt == cell[0] or cotx0_end == cell[0]):
            file = "media/case/{}.jpg".format( str(uuid.uuid4()) )
            hinh = gr[cell[1]:cell[3] , cell[0]:cell[2]]
            
            if cotx0_firt == None or cotx0_end == None:
                to_t= pytesseract.image_to_string(hinh, lang='vie+eng')
                x_macth = re.search(text_content, to_t)
                if x_macth: cotx0_firt = cell[0]
                x_macth = re.search(text_amt, to_t)
                if x_macth: cotx0_end = cell[0]

            elif cotx0_firt == cell[0]:
                to_t= pytesseract.image_to_string(hinh, lang='vie+eng')
                to_text.append(to_t)
            elif cotx0_end == cell[0]:
                to_t= pytesseract.image_to_string(hinh, lang='vie+eng')
                to_value.append(to_t)
    to_value_conver = []
    for value in to_value:
        list = value.split('\n')
        to_value_conver += [i for i in list if i]
    merge_text_value = []
    for key, value in enumerate(to_text):
        merge_text_value.append({
            'text':  value,
            'value': to_value_conver[key]
        })

    return JsonResponse({'data' : merge_text_value})


def min_max_h(img):
    # ảnh nhị phân (binary image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, img_bin = cv2.threshold(
        gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255 - img_bin

    # xác định các cạnh hàng ngang bằng một hàm đơn giản
    kernel_len = gray.shape[1] // 120
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    image_horizontal = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_horizontal, hor_kernel, iterations=3)

    h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi / 180, 30, maxLineGap=250)
    # nhóm những cạnh này lại thành các cạnh chính
    new_horizontal_lines = group_h_lines(h_lines, kernel_len)

    min_l = 9999999
    max_l = 9999999
    for line in new_horizontal_lines:
        y1 = line[1]
        if (min_l > y1 or  max_l == 9999999) and y1 >= 1000 : min_l = y1
        if max_l < y1 or  max_l == 9999999 : max_l = y1
    return min_l, max_l

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def group_h_lines(h_lines, thin_thresh):
    new_h_lines = []
    while len(h_lines) > 0:
        thresh = sorted(h_lines, key=lambda x: x[0][1])[0][0]
        lines = [line for line in h_lines if thresh[1] -
                    thin_thresh <= line[0][1] <= thresh[1] + thin_thresh]
        h_lines = [line for line in h_lines if thresh[1] - thin_thresh >
                    line[0][1] or line[0][1] > thresh[1] + thin_thresh]
        x = []
        for line in lines:
            x.append(line[0][0])
            x.append(line[0][2])
        x_min, x_max = min(x) - int(5*thin_thresh), max(x) + int(5*thin_thresh)
        if int(x_max) - int(x_min) > 1000:
            new_h_lines.append([x_min, thresh[1], x_max, thresh[1]])
    return new_h_lines

def group_v_lines(v_lines, thin_thresh):
    new_v_lines = []
    while len(v_lines) > 0:
        thresh = sorted(v_lines, key=lambda x: x[0][0])[0][0]
        lines = [line for line in v_lines if thresh[0] -
                 thin_thresh <= line[0][0] <= thresh[0] + thin_thresh]
        v_lines = [line for line in v_lines if thresh[0] - thin_thresh >
                   line[0][0] or line[0][0] > thresh[0] + thin_thresh]
        y = []
        for line in lines:
            y.append(line[0][1])
            y.append(line[0][3])
        y_min, y_max = min(y) - int(4*thin_thresh), max(y) + int(4*thin_thresh)
        if int(y_max) - int(y_min) > 200:
            new_v_lines.append([thresh[0], y_min, thresh[0], y_max])
    return new_v_lines

def seg_intersect(line1: list, line2: list):
    a1, a2 = line1
    b1, b2 = line2
    da = a2-a1
    db = b2-b1
    dp = a1-b1

    def perp(a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float))*db + b1

def get_bottom_right(right_points, bottom_points, points):
    for right in right_points:
        for bottom in bottom_points:
            if [right[0], bottom[1]] in points:
                return right[0], bottom[1]
    return None, None

