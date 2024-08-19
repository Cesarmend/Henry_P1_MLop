from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib


# Cargar dataframe
df = pd.read_csv('../Datasets/df_test_output.csv')

# Iniciar FastAPI app
app = FastAPI()



# Endpoint 1: Filmaciones por mes
# Diccionario meses para modulo datetime
months_mapping = {
    'enero': 1,
    'febrero': 2,
    'marzo': 3,
    'abril': 4,
    'mayo': 5,
    'junio': 6,
    'julio': 7,
    'agosto': 8,
    'septiembre': 9,
    'octubre': 10,
    'noviembre': 11,
    'diciembre': 12
}

# Decorador "get" para el numero de peliculas por mes
@app.get("/Cantidad_filmaciones_mes/{mes}", summary="Cantidad de filmaciones por mes")
async def cantidad_filmaciones_mes(mes: str):
    mes = mes.lower()
    if mes not in months_mapping:
        raise HTTPException(status_code=400, detail="Mes no valido")
    
    month_number = months_mapping[mes]
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    month_count = df[df['release_date'].dt.month == month_number].shape[0]
    
    return JSONResponse(content=jsonable_encoder(
        f"{month_count} peliculas han sido lanzadas en {mes.capitalize()}"
    ))

# Endpoint 2: Peliculas por dia
# Diccionario de días en español a sus correspondientes números en el módulo datetime
days_mapping = {
    'lunes': 0,
    'martes': 1,
    'miercoles': 2,
    'jueves': 3,
    'viernes': 4,
    'sabado': 5,
    'domingo': 6
}

# Decorador "get" para la cantidad de películas en un dia en particular
@app.get("/Cantidad_filmaciones_dia/{dia}", summary="Cantidad filmaciones por dia")
async def cantidad_filmaciones_dia(dia: str):
    dia = dia.lower()
    if dia not in days_mapping:
        raise HTTPException(status_code=400, detail="Dia no valido")
    
    day_number = days_mapping[dia]
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    day_count = df[df['release_date'].dt.weekday == day_number].shape[0]
    
    return JSONResponse(content=jsonable_encoder(
        f"{day_count} peliculas han sido lanzadas en {dia.capitalize()}"
    ))

# Endpoint 3: puntaje de la pelicula
# Decorador "get" para año de estreno y puntaje segun la pelicula
@app.get("/Puntaje_por_titulo/{titulo}", summary="Puntaje por titulo")
async def score_titulo(titulo: str):
    # Convertir el título ingresado a minúsculas para evitar problemas de capitalización
    titulo = titulo.lower()
    
    # Filtrar el dataset buscando el título ingresado
    movie = df[df['title'].str.lower() == titulo].to_dict(orient='records')
    
    # Manejo de errores: Si la película no se encuentra, devolver un error 404
    if not movie:
        raise HTTPException(status_code=404, detail="Pelicula no encontrada")
    
    # Extraer la información relevante
    movie_info = movie[0]
    title = movie_info['title']
    release_year = movie_info['release_year']
    score = movie_info['popularity']
    
    # Responder con la información solicitada
    return JSONResponse(content=jsonable_encoder(
        f"La pelicula '{title}' fue estrenada en el año {release_year} con un puntaje de popularidad de {round(score, 2)}"
    ))

# Endpoint 4: pelicula - año - votos y promedio
@app.get("/Votos_titulo/{titulo}", summary="Votos por pelicula")
async def votos_titulo(titulo: str):
    # Convertir el título ingresado a minúsculas para evitar problemas de capitalización
    titulo = titulo.lower()
    
    # Filtrar el dataset buscando el título ingresado
    movie = df[df['title'].str.lower() == titulo].to_dict(orient='records')
    
    # Manejo de errores: Si la película no se encuentra
    if not movie:
        raise HTTPException(status_code=404, detail="Pelicula no encontrada")
    
    # Extraer la información relevante
    movie_info = movie[0]
    vote_count = movie_info['vote_count']
    
    # Verificar si la película tiene al menos 2000 valoraciones
    if vote_count < 2000:
        raise HTTPException(status_code=400, detail="La pelicula no cumple con el minimo de 2000 valoraciones")
    
    title = movie_info['title']
    release_year = movie_info['release_year']
    vote_average = movie_info['vote_average']
    
    # Responder con la información solicitada
    return JSONResponse(content=jsonable_encoder(
        f"La pelicula '{title}' fue estrenada en el año {release_year} con un total de {vote_count} valoraciones y un promedio de {vote_average}"
    ))

# Endpoint 5
@app.get("/Reporte_actor/{nombre_actor}", summary="Reporte financiero por actor")
async def get_actor(nombre_actor: str):
    # Convertir el nombre del actor a minúsculas para evitar problemas de capitalización
    nombre_actor = nombre_actor.lower()
    
    # Filtrar el dataset buscando las películas en las que ha participado el actor
    actor_films = df[df['actor_names'].str.lower().str.contains(nombre_actor, na=False)]
    
    # Manejo de errores: Si no se encuentran películas para ese actor, devolver un error 404
    if actor_films.empty:
        raise HTTPException(status_code=404, detail="Actor no encontrado o no tiene películas en el dataset")
    
    nombre_actor_titulo=nombre_actor.title()
    
    # Calcular la cantidad de películas y los retornos
    film_count = actor_films.shape[0]
    total_return = actor_films['return'].sum()
    average_return = total_return / film_count if film_count > 0 else 0
    
    # Responder con la información solicitada
    return JSONResponse(content=jsonable_encoder(
        f"El actor '{nombre_actor_titulo}' ha participado en {film_count} filmaciones, "
        f"el mismo ha conseguido una ganancia de {total_return:.2f} USD con un promedio de {average_return:.2f} USD por filmación"
    ))

# Endpoint 6

@app.get("/Reporte_director/{nombre_director}", summary="Reporte financiero por director")
async def get_director(nombre_director: str):
    # Convertir el nombre del director a minúsculas para evitar problemas de capitalización
    nombre_director = nombre_director.lower()
    
    # Filtrar el dataset buscando las películas dirigidas por el director
    director_films = df[df['director_name'].str.lower() == nombre_director]
    
    
    # Manejo de errores: Si no se encuentran películas para ese director, devolver un error 404
    if director_films.empty:
        raise HTTPException(status_code=404, detail="Director no encontrado o no tiene peliculas en el dataset")
    
    nombre_director_titulo = nombre_director.title()
    
    # Extraer la información relevante para cada película
    films_info = []
    for _, film in director_films.iterrows():
        film_info = {
            "title": film['title'],
            "release_date": film['release_date'],
            "budget": f"{film['budget']:,}",  # Formato con separadores de miles
            "revenue": f"{film['revenue']:,}",
            "return": round(film['return'], 2)
        }
        films_info.append(film_info)
    
    # Calcular el retorno total del director
    total_return = director_films['return'].sum()
    
    # Responder con la información solicitada
    return JSONResponse(content=jsonable_encoder({
        "director": nombre_director_titulo,
        "total_return": round(total_return, 2),
        "films": films_info
    }))

# Endpoint 7: Machine learning 

key_columns = ['genre_names', 'actor_names', 'overview', 'title']
for col in key_columns:
    df[col] = df[col].fillna('')

# Combinar los atributos para crear el texto a vectorizar
mix_col = df['genre_names'] + ' ' + df['actor_names'] + ' ' + df['title'] + ' ' + df['overview']

# Vectorizar los datos
vectorizer = TfidfVectorizer()
key_vector = vectorizer.fit_transform(mix_col)

# Calcular la similitud del coseno
matriz_simil = cosine_similarity(key_vector)

@app.get("/Recomendacion/{titulo}", summary="Recomendacion de peliculas similares")
async def recomendacion(titulo: str):
    # Convertir el título a minúsculas para evitar problemas de capitalización
    titulo = titulo.lower()

    # Encontrar la película en la lista
    lista_total_peliculas = df['title'].tolist()
    similares = difflib.get_close_matches(titulo, lista_total_peliculas)

    if not similares:
        raise HTTPException(status_code=404, detail="Película no encontrada")

    referencia = similares[0]
    indice_pelicula = df[df['title'] == referencia].index[0]

    # Obtener los índices de las películas similares
    valores_simil = list(enumerate(matriz_simil[indice_pelicula]))
    orden_peliculas_parecidas = sorted(valores_simil, key=lambda x: x[1], reverse=True)

    # Excluir la película original y devolver las 5 más similares
    recomendaciones = []
    for movie in orden_peliculas_parecidas:
        index = movie[0]
        titulo_por_indice = df.iloc[index]['title']
        
        if titulo_por_indice.lower() != titulo:
            recomendaciones.append(titulo_por_indice)
        
        if len(recomendaciones) >= 5:
            break

    if not recomendaciones:
        raise HTTPException(status_code=404, detail="No se encontraron recomendaciones")

    return {"recomendaciones": recomendaciones}

# Endpoint para mostrar tabla de peliculas mas vistas
@app.get("/Tabla_peliculas", summary="(Interfaz) Tabla de peliculas mas vistas")
async def get_table_image():
    return FileResponse("../Images/Tabla_peliculas.png")

# Endpoint para mostrar porcentaje de idiomas
@app.get("/Porcentajes_idiomas", summary="(Interfaz) Distribucion de idiomas")
async def get_bar_chart_image():
    return FileResponse("../Images/idiomas_distribution.png")