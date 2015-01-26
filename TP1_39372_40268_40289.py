# -*- coding: utf-8 -*-
"""
Experiencias para TP1 de PIV

Created on Tue Oct 21 13:29:09 2014

@author: namm
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

print os.getcwd()
fileDir = os.getcwd() #'PIV_14_15_TL1pack/'
fileMoedas1 = 'm1.jpg'
fileMoedas2 = 'm2.jpg'
fileMoedas3 = 'm3.jpg'
fileMoedas4 = 'm4.jpg'
fileMoedas5 = 'm5.jpg'
fileMoedas6 = 'm6.jpg'
fileMoedas7 = 'm7.jpg'
fileMoedas8 = 'm8.jpg'
fileMoedas9 = 'm9.jpg'

files = ['m1.jpg','m2.jpg','m3.jpg','m4.jpg','m5.jpg','m6.jpg','m7.jpg','m8.jpg','m9.jpg']
#img = cv2.imread(fileDir+ '/PIV_14_15_TL1pack/' +  fileMoedas9) 

conjuntoTreino = ['m2.jpg','m3.jpg','m4.jpg','m6.jpg','m7.jpg','m9.jpg']

conjuntoValidacao = ['m1.jpg','m5.jpg','m8.jpg']

conjuntoTotal = ['m1.jpg','m2.jpg','m3.jpg','m4.jpg','m5.jpg','m6.jpg','m7.jpg','m8.jpg','m9.jpg']


def  getRGBImages(src, hist=0, showR=0, showG=0 ,showB=0, showAll=0):
    '''Split the target image into its red, green and blue channels.
  Recebe uma imagem a cores e decompõe-na nas suas componentes de cor R,G e B. 
  Retorna três arrays bidimensionais, cada um correspondente a uma cor. 
  '''    
    red = src[:,:,2]
    r = np.zeros((red.shape[0], red.shape[1], 3),dtype = red.dtype) 
    r[:,:,2] = red
    if(showR == 1 or showAll == 1):
        cv2.imshow('red',r)
        cv2.waitKey(0)
        cv2.destroyAllWindows()    
    green = src[:,:,1]
    g = np.zeros((green.shape[0], green.shape[1], 3),dtype = green.dtype) 
    g[:,:,1] = green
    if(showG == 1 or showAll == 1):
        cv2.imshow('green',g)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    blue = src[:,:,0]
    b = np.zeros((blue.shape[0], blue.shape[1], 3),dtype = blue.dtype) 
    b[:,:,0] = blue
    if (showB == 1 or showAll == 1):
        cv2.imshow('blue',b)
        cv2.waitKey(0)
        cv2.destroyAllWindows()     
    return red,green,blue



#operadores morfologicos
def getImagem( imgName, ploting = 0):
    '''
    Recebe o caminho de uma imagem. Usando a função getRGBImages decompõe a 
    imagem nas três componentes de cor e trabalha sobre a componente de 
    vermelho, aplicando operações morfológicas (blur, erosion e dilate). 
    É retornada a imagem dilatada e a imagem original.
    '''
    img = cv2.imread(fileDir+ '/PIV_14_15_TL1pack/' +  imgName)  
    r,g,b = getRGBImages(img)  
    blur = cv2.medianBlur(r,7,0)
    ret3,bw = cv2.threshold(blur,100,255, cv2.THRESH_BINARY)    
    elemEst = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(14,14))
    erosion = cv2.erode(bw,elemEst,iterations = 3)
    elemEst = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilate = cv2.dilate(erosion,elemEst,iterations = 3)    
    if ploting:
        cv2.imshow('threshold',bw)
        cv2.imshow('Apos erosao',erosion)
        cv2.imshow('Apos dilate',dilate)  
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    return dilate, img

        

def classifyObjects(paths):
    '''
    Recebe um array com um conjunto de caminhos para imagens. Processa-as,
    faz a extração de caracteristicas e classifica-as, plotando a imagem 
    etiquetada cm as classes e o valor total das moedas presentes na imagem.
    '''
    areaLimits = np.array([0,7000,8900, 10500,12250,13500,15200,17000,34000])
    values = np.array([0.01,  0.02,0.1,0.05,0.2,1,0.5,2])
    etiquetas = ["1 cent","2 cent","10 cent","5 cent","20 cent","1 euro","50 cent","2 euro"]
    for p in paths:
        quantia = 0
        imagemBin, imgOriginal = getImagem(p,1)   
        (Contornos, [Hierarquia]) = cv2.findContours(imagemBin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(Contornos)):
            temFilho = (Hierarquia[i][2] != -1)
            temPai = (Hierarquia[i][3] != -1)
            area = cv2.contourArea(Contornos[i])           # Area
            perimetro = cv2.arcLength(Contornos[i], True)       # Perimetro
            circularidade = (perimetro*perimetro)/area  
            momentos = cv2.moments(Contornos[i])
            xCenter = (int) (momentos['m10'] / momentos['m00'])
            yCenter = (int) (momentos['m01'] / momentos['m00'])             
            #se for redondo
            if circularidade>=14 and circularidade<=15 and not (temFilho or temPai):   
                for i in range(len(areaLimits)-1):
                    if area >= areaLimits[i] and area <= areaLimits[i+1]:
                        quantia += values[i]
                        cv2.putText(imgOriginal, etiquetas[i], \
                                    (xCenter-40, yCenter), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0,0,0), \
                                    thickness=1)
        quantia = str(Decimal(quantia).quantize(Decimal('.01')))+'E'
        cv2.putText(imgOriginal, "Quantia total = "+quantia, \
                                    (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, color=(0,0,0), \
                                    thickness=1)
        small = cv2.resize(imgOriginal, (0,0), fx=0.7, fy=0.7)         
        cv2.imshow('Imagem Etiquetada: '+ p,small)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        
    

classifyObjects(conjuntoValidacao)
classifyObjects(conjuntoTreino)