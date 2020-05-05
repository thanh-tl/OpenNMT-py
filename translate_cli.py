from onmt.translate.translation_im import TranslationImCli
import cv2
from PIL import Image
import numpy as np

def convert_to_binary(original_img):

    blur = cv2.GaussianBlur(original_img, (1, 1), 0)

    img_gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

    # https://theailearner.com/tag/cv2-morphologyex/
    # tach va xoa cac diem nhieu
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)

    kernel2 = np.ones((1,1), np.uint8)
    opening = cv2.dilate(opening, kernel2, iterations=1)

    blur = cv2.GaussianBlur(opening,(3,3),0)
    #ret, binary_img = cv2.threshold(img_gray, 120, 1, cv2.THRESH_BINARY_INV)
    ret,binary_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #binary_img = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    # binary_img = binary_img / 255
    # cv2.imshow("binary image", binary_img);

    # cv2.waitKey()
    return binary_img

def remove_contrast(image):
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # img = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    img = cv2.dilate(image,  np.ones((5, 5), np.uint8))
    img = cv2.medianBlur(img, 15)
    img = 255 - cv2.absdiff(image, img)
    norm_img = img.copy()
    cv2.normalize(img, norm_img, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
    norm_img = cv2.threshold(
        norm_img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.normalize(norm_img, norm_img, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.bitwise_not(norm_img, norm_img)
    return norm_img

# img = cv2.imread("/Users/thanhtruongle/Desktop/test_00001.png", cv2.IMREAD_COLOR)
img = cv2.imread("/Users/thanhtruongle/Desktop/test.jpg", cv2.IMREAD_COLOR)
binary_img = remove_contrast(img)

# binary_img = cv2.imread("/Users/thanhtruongle/Desktop/test_00015.png", 0)

cv2.imshow("binary_img", binary_img)
cv2.waitKey()
image = Image.fromarray(binary_img)

c = TranslationImCli()
c.start('./available_models/trans.conf.json')
results, scores, n_best, times, aligns = c.run([{"src":image}])

print("results", results)
print("scores", scores)
print("n_best", n_best)
print("times", times)
print("aligns", aligns)