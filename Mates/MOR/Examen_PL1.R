
library(igraph)
# Enunciado y primeras definiciones ----
A <- matrix(
  c(
    0,0,0,1,0,0,
    0,0,1,0,1,0,
    0,0,0,0,1,0,
    0,0,1,0,0,0,
    0,0,0,1,0,0,
    0,0,0,1,0,0
    ),
  nrow = 6,
  byrow = TRUE
)


G <- graph_from_adjacency_matrix(A, mode = "directed")
plot(G)

# Orden ----
vcount(G)

# Tamaño ----
ecount(G)

# Grado de incidencia ----
degree(G, v = 5, mode = "total") # Grado de incidencia

# Vertices que forman la componente fuertemente conexa que contiene al vertice 5 ----
# Desplegamos los vertices que contiene cada componente y vemos donde esta el cinco y que vertices están con el
comp <- components(G, mode = "strong")
lapply(seq_len(comp$no), function(i) {
  which(comp$membership == i)
})

# Debilmente conexo ----
is_connected(G, mode = "weak")

# Comprobación de si es minimal
is_connected(G) && is_acyclic(G)

# Caminos de longitud 2 entre 2 y 4 ----
A_2 = A %*% A
A_2[2,4] + A_2[4,2]

# Subgrafo y tamaño ----
H <- induced_subgraph(G, vids = c(1,5,6))
plot(H)
ecount(H)

# Comprombacion de si el subgrafo es conexo ----
is.connected(H)

# Orden de las componentes del subgrafo ----
# Como antes desplegamos que vertices existen en cada componente y vemos cuantos son en cada caso
comp2 <- components(H)
lapply(seq_len(comp2$no), function(i) {
  which(comp2$membership == i)
})
