# SRI-Project
## Integrantes:
- Paula Silva Lara C312
- Ricardo Cápiro Colomar C312
- Ariel González Gómez C312

---
## Problema
En el vasto mundo actual de consumo y comercio digital, los usuarios tienen acceso a una cantidad abrumadora de productos, servicios y artículos. Esta amplia variedad, aunque en principio puede parecer una ventaja, presenta un desafío significativo para los usuarios, quienes a menudo se encuentran perdidos en la búsqueda de lo que realmente desean. La dificultad no radica únicamente en encontrar el producto adecuado, sino también en identificar opciones óptimas que se ajusten a sus necesidades, preferencias y recursos personales.

Este problema se ha visto exacerbado por la enorme cantidad de datos generados a diario, lo que ha llevado al desarrollo de los Sistemas de Recomendaciones. Estos sistemas están diseñados para filtrar y presentar productos que se alinean con las preferencias individuales de los usuarios, mejorando así la experiencia de compra y consumo.

Sin embargo, muchos sistemas de recomendación tradicionales enfrentan una limitación fundamental: su visión estática de las preferencias del usuario. Estos sistemas suelen recomendar de manera repetitiva un conjunto fijo de productos basados en datos pasados, sin tener en cuenta la evolución dinámica de los intereses del usuario. Por ejemplo, si un usuario ha comprado una computadora portátil, un sistema tradicional podría seguir recomendando más computadoras portátiles, a pesar de que el interés del usuario puede haber cambiado.

Para abordar esta limitación, los sistemas de recomendación han evolucionado hacia enfoques más sofisticados, como el Filtrado Colaborativo Secuencial. Este enfoque se basa en la idea de que las recomendaciones deben ser contextuales y adaptativas, reflejando la secuencia de acciones del usuario en lugar de ofrecer sugerencias estáticas.

En este contexto, las Redes Neuronales Recurrentes (RNN) juegan un papel crucial. Las RNN están diseñadas para manejar datos secuenciales, lo que las hace particularmente adecuadas para modelar y predecir las preferencias del usuario basándose en el historial de acciones. A diferencia de los enfoques tradicionales, que tratan cada recomendación de forma aislada, las RNN consideran la secuencia completa de interacciones del usuario. Esto permite al sistema identificar patrones y cambios en las preferencias a lo largo del tiempo, proporcionando recomendaciones más precisas y relevantes.

El objetivo de este proyecto es desarrollar un Sistema de Recomendación Secuencial, utilizando una RNN para mejorar la precisión y adaptabilidad de las recomendaciones. Este sistema permitirá a los usuarios registrados recibir sugerencias personalizadas sobre los productos que probablemente les interesen, basadas no solo en sus preferencias pasadas, sino también en el contexto de sus acciones más recientes. Así, se superarán las limitaciones de los sistemas tradicionales y se ofrecerá una experiencia de recomendación más dinámica y acorde a las necesidades cambiantes de los usuarios.

---

## Requerimientos

### Hardware:

Las Redes Neuronales Recurrentes (RNN) son modelos complejos que requieren una considerable cantidad de cálculos para su entrenamiento. Dado que la base de datos incluye 16 490 productos diferentes, cada uno con múltiples características: ID, marca, precio y tipo de producto, y  un historial extenso de interacciones de los usuarios, el entrenamiento del modelo se vuelve significativamente intensivo en términos de procesamiento. Las CPUs tradicionales, pueden enfrentar tiempos de entrenamiento prolongados debido a la alta demanda computacional asociada con el procesamiento de grandes volúmenes de datos y la actualización constante de los pesos del modelo durante el entrenamiento. 
<br></br>
La aceleración proporcionada por una GPU, es crucial para reducir significativamente los tiempos de entrenamiento y mejorar el rendimiento general del sistema.
<br></br>
El proyecto ha sido ejecutado en Google Colab, que proporciona acceso a una GPU gratuita para el procesamiento de datos.

## Origen de los datos

https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store

Esta base de datos proporciona las compras que han llevado a cabo usuarios con las respectivas características de los productos comprados. 

Es seleccionado el archivo que contiene las compras de noviembre del 2019. 
<br></br>
La base de datos usada en el proyecto es una versión reducida de esta, proporcionada en el archivo [eCommerce.csv](../src/eCommerce.csv), obtenida mediante el código [process_data.py](../src/process_data.py).

## Uso
Para iniciar el proyecto, ejecuta:
```bash
chmod +x startup.sh
./startup.sh

```
