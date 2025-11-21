from scipy.ndimage import convolve
import numpy as np
import skimage
from skimage import io
from skimage import morphology 
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os

def ombres(img):

    """
    :param img : l'image RGB  
    :return: Un masque binaire obtenu via un seuillage grossier (pour obtenir les zones zombres)
    """
    
    img_gray = skimage.color.rgb2gray(img)
    thresh = threshold_otsu(img_gray)
    binary_img = img_gray < thresh
    radius = 10
    binary_img = morphology.binary_erosion(image=binary_img, footprint=morphology.disk(radius))
    binary_img = morphology.binary_dilation(image=binary_img, footprint=morphology.rectangle(50,5)) 
    binary_img = morphology.binary_dilation(image=binary_img, footprint=morphology.rectangle(5,50))
    return binary_img    #Boolean

def get_mask(img):

  """
  :param img : l'image RGB  
  :return: Un masque (disque) correspondant à la zone sans bordure (Matrice de 1 si pas de bordure)
  """
    
  nlin, ncol, _ = np.shape(img)

  #On forme un masque pour le recadrage
  binary_img = ombres(img)

  count = 0
  i = 0
  while(binary_img[i][i]):
    count += 1
    i += 1 
  r1 = count

  count = 0
  i = 1
  while(binary_img[-i][-i]):
    count += 1
    i += -1 
  r2 = count

  count = 0
  i, j = nlin-1,0
  while(binary_img[i][j]):
    count += 1
    j +=1
    i += -1 
  r3 = count

  count = 0
  i,j = 0, ncol-1
  while(binary_img[i][j]):
    count += 1
    i += 1
    j += -1 
  r4 = count

  Lmax = max([r1,r2,r3,r4])

  antibordure = np.ones((nlin,ncol))
  antibordure[:10, :] = 0
  antibordure[:, :10] = 0
  antibordure[nlin-10:, :] = 0
  antibordure[:, ncol-10:] = 0
  antibordure = antibordure > 0

  if Lmax == 0: #Cela correspond au cas où il n'y a pas de bordure/zone d'ombre dans les coins.
    return antibordure

  centre = np.array([nlin//2, ncol//2]) # Centre du disque (et de l'image)
  ptL = np.array([Lmax, Lmax]) # point par lequel le cercle doit passer
  deplacement = centre - ptL
  rayon = np.linalg.norm(deplacement)
  
  disk_mask = morphology.disk(rayon - 10) # le - 10 est pour être bien sûr

  maxn = max(nlin, ncol)
  new = np.zeros((2*maxn, 2*maxn))
  ds = disk_mask.shape
  new[int(maxn - rayon) : int(maxn - rayon) + ds[0], int(maxn - rayon) : int(maxn - rayon) + ds[1]] = disk_mask

  recadrage = np.zeros((nlin,ncol))
  recadrage = new[maxn - nlin//2 : maxn - nlin//2 + nlin, maxn - ncol//2 : maxn - ncol//2 + ncol]

  return recadrage * antibordure

def apply_mask(img, mask):
  
  """
  :param img : l'image 
  :param mask : le masque à appliquer
  :return: l'image recadrée
  """
    
  h, w = img.shape[:2]
  img_mask = np.copy(img)
  for i in range(h):
    for j in range(w):
      if not mask[i][j]:
        img_mask[i][j] = 0
  return img_mask

def matmax(img):

  """
  :param img : l'image RGB
  :return: tuple des masques (pour R, G et B de l'image) des pixels de poils
  """

  R = img[:,:,0]
  G = img[:,:,1]
  B = img[:,:,2]

  s0 = morphology.rectangle(50,10)
  s1 = morphology.rectangle(10,50)

  r0 = morphology.closing(R, footprint=s0)
  r1 = morphology.closing(R, footprint=s1)

  g0 = morphology.closing(G, footprint=s0)
  g1 = morphology.closing(G, footprint=s1)

  b0 = morphology.closing(B, footprint=s0)
  b1 = morphology.closing(B, footprint=s1)

  matmaxr = np.zeros(R.shape)
  matmaxg = np.zeros(R.shape)
  matmaxb = np.zeros(R.shape)
  nlin, ncol = R.shape

  for i in range(nlin):
    for j in range(ncol):
      matmaxr[i,j] = max(r0[i,j], r1[i,j])
      matmaxg[i,j] = max(g0[i,j], g1[i,j])
      matmaxb[i,j] = max(b0[i,j], b1[i,j])

  return (matmaxr, matmaxg, matmaxb)

def hair_mask(img, matmax):

  """
  :param img : l'image RGB
  :param matmax: tuple des trois masques de poils R, G et B
  :return: tuple des masques (pour R, G et B de l'image) des pixels de poils (avec des zones de poils élargies)
  """

  R = img[:,:,0]
  G = img[:,:,1]
  B = img[:,:,2]

  matmaxr, matmaxg, matmaxb = matmax
  Mr = np.zeros(matmaxr.shape)
  Mg = np.zeros(matmaxr.shape)
  Mb = np.zeros(matmaxr.shape)

  Mr[:,:] = np.abs(R[:,:]-matmaxr[:,:]) > 40
  Mg[:,:] = np.abs(G[:,:]-matmaxg[:,:]) > 40
  Mb[:,:] = np.abs(B[:,:]-matmaxb[:,:]) > 40

  square = morphology.rectangle(20, 20)
  Mr = morphology.binary_dilation(Mr, square)
  Mb = morphology.binary_dilation(Mb, square)
  Mg = morphology.binary_dilation(Mg, square)

  return (Mr, Mg, Mb)

#FONCTION PRISE SUR INTERNET 
def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image

def hairRemoval(img, mask):

  """
  :param img : l'image RGB
  :return: l'image sans les poils
  """
  
  matmax_ = matmax(img)
  (matmaxr, matmaxg, matmaxb) = matmax_
  (Mr, Mg, Mb) = hair_mask(img, matmax_)

  hairDetection = morphology.erosion(np.abs(img[:,:,0]-matmaxr), footprint=morphology.disk(12))
  if hairDetection.sum() / mask.sum() < 0.2:
    return img

  R = img[:,:,0]
  G = img[:,:,1]
  B = img[:,:,2]

  newR = interpolate_missing_pixels(R, Mr > 0)
  newB = interpolate_missing_pixels(B, Mb > 0)
  newG = interpolate_missing_pixels(G, Mg > 0)

  square = morphology.rectangle(20, 20)
  Rf = skimage.filters.median(newR, square)
  Gf = skimage.filters.median(newG, square)
  Bf = skimage.filters.median(newB, square)

  final_img = np.copy(img)
  final_img[:,:,0]=Rf
  final_img[:,:,1]=Gf
  final_img[:,:,2]=Bf

  return final_img
