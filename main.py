import sys
import cv2,numpy as np,pandas as pd,skvideo.io,os,random
from jeanCV import skinDetector
from functions import *
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
	cap = []
	if len(sys.argv) == 3:
		try:
			cap = skvideo.io.vread(sys.argv[2])
		except:
			print("No pudimos leer el video: "+str(sys.argv[2]))

		column_min_capa_verde = []
		column_max_capa_verde = []
		column_mean_capa_verde = []
		column_std_capa_verde = []
		faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt0.xml")
		faceCascade_profileface = cv2.CascadeClassifier("haarcascades/haarcascade_profileface.xml")
		faceCascade_default = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
		faceCascade1 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt1.xml")
		faceCascade2 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
		alphas = [1]
		betas = [100]
		scale_factor = 1.3
		minneighbours = 5
		len_total = cap.shape[0]
		index = 0
		for img in cap:
			gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			roi_frame = img
			face_rects = faceCascade.detectMultiScale(gray, scale_factor, minneighbours)

			if len(face_rects) == 1:
				for (x, y, w, h) in face_rects:
					roi_frame = img[y:y + h, x:x + w]
				detector = skinDetector(cv2.cvtColor(np.array(roi_frame), cv2.COLOR_BGR2RGB))
				roi_frame = detector.find_skin()
				roi_frame = roi_frame[:,:,:]
				frame = np.ndarray(shape=roi_frame.shape, dtype="float")
				frame[:] = roi_frame * (1./255)
				capa_verde = np.zeros((frame.shape[0],frame.shape[1],1))
				capa_verde[:,:,0] = frame[:,:,1]
				capa_verde = capa_verde.reshape(frame.shape[0]*frame.shape[1])
				del frame;
				pixeles = []
				for a in capa_verde:
					if a > 0:
					  pixeles.append(a)
				min = np.min(pixeles)*255
				max = np.max(pixeles)*255
				mean = np.mean(pixeles)*255
				std = np.std(pixeles)*255
				column_min_capa_verde.append(min)
				column_max_capa_verde.append(max)
				column_mean_capa_verde.append(mean)
				column_std_capa_verde.append(std)
			index += 1

		if len(column_min_capa_verde) == 0:
			print("Lo siento, no pudimos detectar un rostro")
		df_res = pd.DataFrame()
		df_res["min_capa_verde"] = pd.Series(column_min_capa_verde)
		df_res["max_capa_verde"] = pd.Series(column_max_capa_verde)
		df_res["mean_capa_verde"] = pd.Series(column_mean_capa_verde)
		df_res["std_capa_verde"] = pd.Series(column_std_capa_verde)    
		min, max, mean, std, pred = predecir_senial_hr(df_res)
		print("HR estimation: "+str(pred))
	else:
		print("Por favor, ingrese correctamente los parametros")