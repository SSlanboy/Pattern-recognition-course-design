import skimage
from skimage.feature import hog
from skimage import io
from PIL import Image
import cv2
import warnings
# img = cv2.cvtColor(cv2.imread('../img/test.jpg'), cv2.COLOR_BGR2GRAY)
# img = cv2.cvtColor(cv2.imread('lena.png'), cv2.COLOR_BGR2GRAY)
# print(img.shape)
# normalised_blocks, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8),
#                                    block_norm='L2-Hys', visualize=True)
#
# # print(type(hog_image))
# #
# # io.imshow(hog_image)
# # io.show()
#
#
# img = cv2.cvtColor(cv2.imread('1.jpg'), cv2.COLOR_BGR2GRAY)
#
# # io.imshow(img)
# # io.show
#
# normalised_blocks, hog_image = hog(img, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(4, 4),
#                                    block_norm='L2-Hys', visualize=True)
#
# # print(type(hog_image))
#
# # io.imshow(hog_image)
# io.show()
# Image.open(hog_image)


from skimage.feature import hog
from skimage import io
from PIL import Image
import cv2

# from PyInstaller.utils.hooks import collect_data_files, collect_submodules
# datas = collect_data_files("skimage.io._plugins")
# hiddenimports = collect_submodules("skimage.io._plugins")

img = cv2.cvtColor(cv2.imread('Lena.png'), cv2.COLOR_BGR2GRAY)
# print
# img.shape
normalised_blocks, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8),
                                   block_norm='L2-Hys', visualize=True)
io.imshow(hog_image)
io.show()

warnings.filterwarnings("ignore")
