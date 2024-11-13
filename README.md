- Entrenamiento:
  python
  from sklearn.cluster import KMeans

  kmeans = KMeans(n_clusters=k)
  kmeans.fit(X) # X es tu conjunto de características seleccionadas
  
 Clustering Jerárquico

- Configuración:
  - Decide el método de enlace (linkage) a utilizar, como 'ward', 'complete', 'average', etc.
  
- Entrenamiento:
  python
  from scipy.cluster.hierarchy import dendrogram, linkage

  linked = linkage(X, method='ward')
  

 Evaluación del Desempeño del Modelo

Utiliza métricas para evaluar ambos modelos:

- Coeficiente de Silhouette:
  python
  from sklearn.metrics import silhouette_score
  
  silhouette_kmeans = silhouette_score(X, kmeans.labels_)
  
  Para clustering jerárquico
  from sklearn.cluster import AgglomerativeClustering
  
  hierarchical = AgglomerativeClustering(n_clusters=k)
  hierarchical.fit(X)
  
  silhouette_hierarchical = silhouette_score(X, hierarchical.labels_)
  

- Índice de Calinski-Harabasz:
python
from sklearn.metrics import calinski_harabasz_score

ch_kmeans = calinski_harabasz_score(X, kmeans.labels_)
ch_hierarchical = calinski_harabasz_score(X, hierarchical.labels_)


Visualización de Resultados

Realiza gráficas para visualizar los resultados:

- K-means:
python
import matplotlib.pyplot as plt

plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.title("K-means Clustering")
plt.show()


- Clustering Jerárquico:
python
dendrogram(linked)
plt.title("Dendrogram for Hierarchical Clustering")
plt.show()


 Interpretación y Documentación

Finalmente, documenta tus hallazgos:

- Analiza qué significan los clústeres obtenidos. ¿Qué características tienen en común?
- Discute la calidad del modelo usando las métricas obtenidas.
- Haz un resumen en un informe claro que explique cada paso y tus conclusiones.

