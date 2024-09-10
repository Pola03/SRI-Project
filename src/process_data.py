import pandas as pd

chunk_size = 100000  

# Crea un iterador que lea el archivo CSV en chunks
chunks = pd.read_csv('2019-Nov.csv', chunksize=chunk_size)

# Filtra y guarda las filas en las que el tipo de evento sea compra
filtered_chunks = []
for chunk in chunks:
    filtered_chunk = chunk[chunk['event_type'] == 'purchase']
    filtered_chunks.append(filtered_chunk)

# Concatenar todos los chunks filtrados
df = pd.concat(filtered_chunks)

del df['event_time']