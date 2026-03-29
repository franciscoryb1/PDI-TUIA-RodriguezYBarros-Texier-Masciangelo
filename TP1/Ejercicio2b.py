import cv2
import numpy as np
import matplotlib.pyplot as plt

# def correccion(multiple_choice):
    
#     img = cv2.imread('TP1/multiple_choice_{}.png'.format(multiple_choice),cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('multiple_choice_3.png',cv2.IMREAD_GRAYSCALE)

def correccion(img1):
   
   #plt.imshow(img1, cmap='gray')
   #plt.show()

# Vamos a binarizar la img haciendo que sea solo blanco o negro para que despues podamos trabajar mejor
# Los niveles por debajo de 244 van a ser negro y el resto blanco 
# Elegimos estos umbrales probando. Si elegiamos el umbral mas alto quedaban algunos pixeles 
#al rededor de los circulos de las letras con valores 250-254 que quedaban afuera del rango
    _, img_binarizada = cv2.threshold(img1, 244, 255, cv2.THRESH_BINARY_INV)
#plt.imshow(img_binarizada, cmap='gray')
#plt.show()

# Utilizo la img binarizada porque en la img original hay algunos valores que causan problemas 
#(dividen el circulo en algunos puntos), y de esta forma no

    rec = img_binarizada[160:, :]
    #plt.imshow(rec, cmap='gray')
    #plt.show(block=False)
    np.unique(rec)

    # Identifico las Filas
    img_binarizada_row_zeros = rec.any(axis=1)
    img_binarizada_row_zeros_idxs = np.argwhere(rec.any(axis=1))


    xr = np.arange(rec.shape[0])
    yr = img_binarizada_row_zeros*(rec.shape[1]-1)
    #plt.plot(yr, xr, c='r')
    #plt.show(block=False) 

    x = np.diff(img_binarizada_row_zeros)
    renglones_indxs = np.argwhere(x) 
    len(renglones_indxs)


    # * Modifico índices *****
    ii = np.arange(0,len(renglones_indxs),2)
    renglones_indxs[ii]+=1

    # Visualizo
    xri = np.zeros(rec.shape[0])
    xri[renglones_indxs] = (rec.shape[1]-1)
    yri = np.arange(rec.shape[0])            
    #plt.figure(), plt.imshow(rec, cmap='gray'), plt.plot(xri, yri, 'r'), plt.title("Renglones - Inicio y Fin"), plt.show(block=False) 

    # Re-ordeno los índices en grupos de a 2 (inicio-final)
    #x_indxs = renglones_indxs[:(len(renglones_indxs)//2)*2]
    #x_indxs = x_indxs.reshape((-1,2))
    r_idxs = np.reshape(renglones_indxs, (-1,2)) 


    # Obtengo renglones 
    renglones = []
    for ir, idxs in enumerate(r_idxs):
        renglon_img = rec[idxs[0]:idxs[1], :]
        renglones.append({
            "ir": ir+1,
            "cord": idxs,
            "img": rec[idxs[0]:idxs[1],:]
        })

    # plt.figure()
    # for ii, renglon in enumerate(renglones):
    #     # plt.subplot(2,2,ii+1)
    #     plt.figure()
    #     plt.imshow(renglon["img"], cmap='gray')
    #     plt.title(f"Renglón {ii+1}")
    # plt.show(block=False)   


    respuestas_correctas = ['A', 'A', 'B', 'A', 'D', 'B', 'B', 'C', 'B', 'A', 
                            'D', 'A', 'C', 'C', 'D', 'B', 'A', 'C', 'C', 'D', 
                            'B', 'A', 'C', 'C', 'C']

    # Lista para almacenar los resultados
    resultados = []

    # Contador para saber cuantas respuestas correctas hay 
    resp_corr_cont = 0


    # Procesar cada línea
    for ii,renglon in enumerate(renglones):
        img = renglon["img"]
    # 1. Detectar los contornos de los círculos
        contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 2. Asignar índices a los círculos
        circulos = []
        for i, contorno in enumerate(contornos):
            # Filtrar solo contornos con un área mínima
            area = cv2.contourArea(contorno)
            if area > 100:
                circulos.append(contorno)

        # Lista para almacenar la cantidad de píxeles blancos en cada círculo
        pixeles_blancos = []

        # 3. Contar píxeles blancos dentro de cada círculo
        opciones = ['E', 'D', 'C', 'B', 'A']
        pixeles_blancos = []
        for circulo in circulos:
            # Obtener el rectángulo delimitador del círculo
            x, y, w, h = cv2.boundingRect(circulo)
            
            # Recortar el círculo
            circulo_recortado = img[y:y+h, x:x+w]
            
            # Contar píxeles blancos
            pixeles_blancos.append(np.sum(circulo_recortado == 255))

        # Verificar si ninguna opción está seleccionada
        # la cantidad de pixeles blancos que tiene una opcion sin seleccionar son: 
        #{E:179, D:167, C:164, B:187, A:195}, por eso usamos 200
        if all(cantidad < 200 for cantidad in pixeles_blancos):
            resultados.append(f"Pregunta {ii+1}: MAL")
            # print("¡Error! Ninguna opción seleccionada en la línea", i+1)
        else:
            # Imprimir la cantidad de píxeles blancos en cada círculo
            # for i, cantidad in enumerate(pixeles_blancos):
            #     print(f"Línea {i+1}, Círculo {opciones[i]} tiene {cantidad} píxeles blancos.")

            # Verificar si hay más de una opción seleccionada
            # cantidad_seleccionadas = sum(cantidad > 0 for cantidad in pixeles_blancos)
            suma_pixeles = sum(pixeles_blancos)
            if  suma_pixeles > 1100:
                resultados.append(f"Pregunta {ii+1}: MAL")
                # print("¡Error! Más de una opción seleccionada en la línea", i+1)
            else:
                # Identificar la opción seleccionada
                indice_opcion_seleccionada = np.argmax(pixeles_blancos)
                opcion_seleccionada = opciones[indice_opcion_seleccionada]
                if opcion_seleccionada == respuestas_correctas[ii]:
                    resultados.append(f"Pregunta {ii+1}: OK")
                    resp_corr_cont += 1
                else:
                    resultados.append(f"Pregunta {ii+1}: MAL")
                # print("La opción seleccionada en la línea", i+1, "es:", opcion_seleccionada)

    # Imprimir resultados
    for resultado in resultados:
        print(resultado)

#ejercicio 2 A
correccion(img1)


#ejercicio 2 B
def analizar_imagen_rasgos_letras(img):
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Encontrar contornos en la imagen umbralizada
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Dibujar los contornos en una imagen en blanco
    contour_img = np.zeros_like(img)
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Detecta los componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    stats = sorted(stats, key=lambda x: x[0])

    # Filtra los componentes cuya área es menor a 50 pixeles
    stats_filtrado = [s for s in stats if s[-1] < 50]

    umbral_max = 9
    espacios_entre_letras = 0

    repetidos = []

    for i in range(len(stats_filtrado)-1):
        # Obtiene las coordenadas x del componente actual y del siguiente
        x_actual = stats_filtrado[i][0]
        x_siguiente = stats_filtrado[i + 1][0]
        # Calcula la distancia horizontal entre los componentes
        distancia_horizontal = x_siguiente - x_actual

        # Si la distancia horizontal es mayor que cierto umbral intuimos que hay un espacio entre palabras.
        if distancia_horizontal >= umbral_max:
            # Incrementa el contador de espacios entre letras
            espacios_entre_letras += 1

    # Elimina los elementos de "stats_filtrados" que están en "repetidos"
    stats_filtrado = [arr for arr in stats_filtrado if not any((arr == elem).all() for elem in repetidos)]

    salida = {
        "Caracteres": len(stats_filtrado),
        "Espacios": espacios_entre_letras,
        "Palabras": espacios_entre_letras + 1
    }

    return salida

examenes = ["multiple_choice_1.png",
            "multiple_choice_2.png",
            "multiple_choice_3.png",
            "multiple_choice_4.png",
            "multiple_choice_5.png"]

for examen in examenes:
    # Cargo la imagen
    img = cv2.imread(examen,cv2.IMREAD_GRAYSCALE)
    # Corto el encabezado
    encabezado = img[108:130, 30:750]
    # Corto cada campo
    nombre = encabezado[0:20, 70:250]
    id = encabezado[0:20, 300:400]
    code = encabezado[0:20, 460:535]
    fecha = encabezado[0:20, 619:725]
    # Obtengo los datos de cada campo
    d_nombre = analizar_imagen_rasgos_letras(nombre)
    d_id = analizar_imagen_rasgos_letras(id)
    d_code = analizar_imagen_rasgos_letras(code)
    d_fecha = analizar_imagen_rasgos_letras(fecha)

    print("EXAMEN:", examen)
    #Imprimo d_nombre
    print("- NOMBRE:")
    print("--- CARACTERES:", d_nombre["Caracteres"])
    print("--- PALABRAS:", d_nombre["Palabras"])

    #Imprimo d_id
    print("- ID:")
    print("--- CARACTERES:", d_id["Caracteres"])
    print("--- PALABRAS:", d_id["Palabras"])

    #Imprimo d_code
    print("- CODE:")
    print("--- CARACTERES:", d_code["Caracteres"])
    print("--- PALABRAS:", d_code["Palabras"])

    #Imprimo d_fecha
    print("- FECHA:")
    print("--- CARACTERES:", d_fecha["Caracteres"])
    print("--- PALABRAS:", d_fecha["Palabras"])


    #ejercicio c
#print(f"El examen tuvo {resp_corr_cont} respuestas correctas")
#print("Examen APROBADO" if resp_corr_cont >= 20 else "Examen DESAPROBADO")