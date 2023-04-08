# InteliPhsi

Buenos dias Sebas, este codigo es uno de mis muchos intetnos de retomar el entrenar un modelo para que 
me distinga entre phishings y url normales, el fichero importante es el de GPU, no porque la use (que es la idea pero no la realidad aun) si no porque es 
la version que está alineada con el fichero requisitos.txt, si la quieres probar, lo que vas a ver es como se entrena la ia y te la guarda.

Mas tarde implemetaré pruebas pero solo cuando suba la preciosion a mas de 80% (tengo 36% ahora)

     precision    recall  f1-score   support

           0       0.41      0.61      0.49       236
           1       0.24      0.12      0.16       240

    accuracy                           0.36       476
   macro avg       0.32      0.37      0.32       476
weighted avg       0.32      0.36      0.32       476


El 0 son lo bueno que es detectando falsos y el 1 positivos, como ves está chungo
