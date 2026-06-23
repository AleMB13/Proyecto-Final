# Librerias
import os
import pickle
import chromadb
from chromadb.utils import embedding_functions

def inicializar_y_cargar_vector_db():
    # Definir rutas relativas (funcionan en cualquier computadora clonada de GitHub)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH = os.path.join(BASE_DIR, "chroma_unalm_db")
    PKL_PATH = os.path.join(BASE_DIR, "data", "chunks_seleccionados.pkl")
    COLLECTION_NAME = "actas_fep_unalm"

    print(f"-> Inicializando cliente persistente en: {DB_PATH}")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    # Reiniciar colección para evitar duplicados en desarrollo
    if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(name=COLLECTION_NAME)

    # Crear colección con Similitud de Coseno (Métrica obligatoria a justificar)
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} 
    )

    # Configurar modelo de embeddings local
    embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Cargar los datos reales del archivo pkl
    if not os.path.exists(PKL_PATH):
        raise FileNotFoundError(f"No se encontró el archivo pickle en la ruta: {PKL_PATH}. Verifica que Persona 1 lo haya subido.")
        
    with open(PKL_PATH, "rb") as f:
        chunks_reales = pickle.load(f)

    print(f"-> Cargados {len(chunks_reales)} chunks desde el repositorio.")

    documents_list = []
    metadatas_list = []
    ids_list = []

    for i, doc in enumerate(chunks_reales):
        contenido_texto = doc.get('page_content', doc.get('text', ''))
        metadatos_originales = doc.get('metadata', {})
        
        documents_list.append(contenido_texto)
        
        # Limpieza para asegurar compatibilidad estricta con ChromaDB
        clean_metadata = {}
        for key, val in metadatos_originales.items():
            if isinstance(val, (str, int, float, bool)):
                clean_metadata[key] = val
            else:
                clean_metadata[key] = str(val)
                
        metadatas_list.append(clean_metadata)
        ids_list.append(f"chunk_{i}")

    # Indexar en bloques pasándole el modelo de embeddings
    collection.add(
        documents=documents_list,
        metadatas=metadatas_list,
        ids=ids_list,
        embedding_function=embedding_model
    )

    print(f"¡Éxito! Vectores creados e indexados en '{DB_PATH}'. Total: {collection.count()}")
    return collection

if __name__ == "__main__":
    # Esto permite ejecutar el script directamente desde la terminal: python 03_vector_db.py
    inicializar_y_cargar_vector_db()
