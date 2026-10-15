# PRACTICA 1 - INTRODUCCION A LA TEORIA DE GRAFOS
# Modelos de Optimizacion en Redes

# 1.1. PAQUETE igraph ----
# Ejecutar SOLO la primera vez:
# install.packages("igraph")

library(igraph)

# 1.2. INTRODUCCION A GRAFOS ----

## 1.2.1. Primeros pasos para grafos no orientados ----
G <- graph(
  c(1,2,
    1,3,
    1,4,
    2,4,
    4,5,
    4,6,
    5,6,
    7,8),
  directed = FALSE
)

plot(G)

plot(
  G,
  layout = layout_with_fr(G),
  vertex.size = 30,
  vertex.color = "lightblue",
  vertex.label.cex = 1,
  edge.arrow.size = 0.5
)

plot(G, layout = layout_in_circle(G))
plot(G, layout = layout_with_kk(G))
plot(G, layout = layout_as_tree(G))

# 1.2.2. Primeros pasos para grafos orientados
G2 <- graph(
  c(1,3,
    3,1,
    1,2,
    2,4,
    4,1,
    4,5,
    5,6,
    4,6,
    7,8),
  directed = TRUE
)

plot(G2)

is_directed(G)
is_directed(G2)

# 1.3. CONCEPTOS BASICOS -----


## 1.3.1. Orden y tamano ----
vcount(G)
vcount(G2)

ecount(G)
ecount(G2)

## 1.3.2. 1-grafos ----
!any_multiple(G)
!any_multiple(G2)

Gbis <- add_edges(G, c(1, 2))

G_extra <- add_edges(G, c(1, 2, 2, 4, 4, 2))
plot(G_extra)

G2_extra <- add_edges(G2, c(1, 2, 2, 4, 4, 2))

!any_multiple(G_extra)
!any_multiple(G2_extra)

max(count_multiple(G_extra))
max(count_multiple(G2_extra))

id <- get_edge_ids(G, c(1,2))
G_sin_arista <- delete_edges(G, id)
plot(G_sin_arista)

## 1.3.3. Bucles ----
any_loop(G)
any_loop(G2)

G_bucle <- add_edges(G, c(1, 1, 3, 3))
plot(G_bucle)

any_loop(G_bucle)

sum(which_loop(G_bucle))

## 1.3.4. Grafo simple----
is_simple(G)
is_simple(G2)
is_simple(G_extra)
is_simple(G2_extra)
is_simple(G_bucle)

## 1.3.5. Grafo orientado equivalente ----
G_orientado <- as_directed(G)
is_directed(G_orientado)
plot(G_orientado)

## 1.3.6. Grafo subyacente no orientado ----
G2_no_orientado <- as_undirected(G2, mode = "collapse")
is_directed(G2_no_orientado)
plot(G2_no_orientado)


# 1.4. GRAFOS ORIENTADOS ----

## 1.4.1. Grafos completos ----
K5_dir <- make_full_graph(5, directed = TRUE)

## 1.4.2. Subgrafo generado ----
H <- induced_subgraph(G2, vids = c(1,2,3,4))
H <- induced_subgraph(G2, vids = 1:4)

## 1.4.3. Grafo parcial ---- 
ids <- get.edge.ids(G2, c(1,2, 1,3, 5,6))
H2 <- subgraph.edges(G2, ids, delete.vertices = FALSE)

## 1.4.4. Grafo complementario ----
G2c <- complementer(G2, loops = FALSE)

L <- layout_in_circle(G2)
par(mfrow = c(1,2))
plot(G2, layout = L, main = "G2")
plot(G2c, layout = L, main = expression(G2^c))
par(mfrow = c(1,1))

## 1.4.5. Grados de incidencia en un grafo orientado ----
degree(G2, mode = "out")
degree(G2, mode = "in")
degree(G2, mode = "total")

degree(G2, v = 4, mode = "out")
degree(G2, v = 4, mode = "in")
degree(G2, v = 4, mode = "total")

## 1.4.6. Sucesores y antecesores ----
neighbors(G2, 4, mode = "out")
neighbors(G2, 4, mode = "in")
neighbors(G2, 4, mode = "all")

## 1.4.7. Ascendientes y descendientes ----
subcomponent(G2, 2, mode = "out")
subcomponent(G2, 5, mode = "in")

setdiff(as_ids(subcomponent(G2, 3, mode = "out")), 3)
setdiff(as_ids(subcomponent(G2, 3, mode = "in")), 3)

# 1.5. GRAFOS NO ORIENTADOS ----

## 1.5.1. Grafos completos ----
K6 <- make_full_graph(6, directed = FALSE)

## 1.5.2. Subgrafo generado, grafo parcial y complementario ----
H3 <- induced_subgraph(G, vids = c(1,2,3,4))

ids2 <- get.edge.ids(G, c(1,2, 7,8))
H4 <- subgraph.edges(G, ids2, delete.vertices = FALSE)

Gc <- complementer(G, loops = FALSE)

## 1.5.3. Grado de incidencia en un grafo no orientado ----
degree(G)
degree(G, v = 3)

# 1.6. REPRESENTACION MATRICIAL ----

A2 <- as_adjacency_matrix(G2, sparse = FALSE)
A2

A <- as_adjacency_matrix(G, sparse = FALSE)
A

## 1.6.1. Numero de caminos/cadenas de longitud fija ----
A2_4 <- A2 %*% A2 %*% A2 %*% A2
A2_4[1,3]

A_4 <- A %*% A %*% A %*% A
A_4[1,3] + A_4[3,1]

## 1.6.2. Construccion de un grafo a partir de su matriz de adyacencia ----
G2bis <- graph_from_adjacency_matrix(
  A2,
  mode = "directed"
)

plot(G2bis)

Gbis <- graph_from_adjacency_matrix(
  A,
  mode = "undirected"
)

plot(Gbis)

# 1.7. CONEXION EN GRAFOS NO ORIENTADOS ----

is_connected(G)

components(G)
components(G)$membership
components(G)$csize
components(G)$no

comp <- components(G)

lapply(seq_len(comp$no), function(i) {
  which(comp$membership == i)
})

A3 <- matrix(
  c(
    0, 1, 1, 1, 1,
    1, 0, 0, 0, 0,
    1, 0, 0, 0, 0,
    1, 0, 0, 0, 0,
    1, 0, 0, 0, 0
  ),
  nrow = 5,
  byrow = TRUE
)

G3 <- graph_from_adjacency_matrix(A3, mode = "undirected")
is_connected(G3) && is_acyclic(G3)

bridges(G3)

# 1.8. CONEXION EN GRAFOS ORIENTADOS----


## 1.8.1. Grafos fuertemente conexos----
is_connected(G2, mode = "strong")

comp <- components(G2, mode = "strong")

comp$no
comp$csize
comp$membership

lapply(seq_len(comp$no), function(i) {
  which(comp$membership == i)
})

## 1.8.2. Grafos debilmente conexos----
is_connected(G2, mode = "weak")

## 1.8.3. Grafos cuasifuertemente conexos----
any(sapply(V(G2), function(v)
  length(subcomponent(G2, v, mode = "out")) == vcount(G2)
)) || any(sapply(V(G2), function(v)
  length(subcomponent(G2, v, mode = "in")) == vcount(G2)
))

# 1.9. CASO PRACTICO RESUELTO: ANALISIS DE UNA RED SOCIAL ----


G_social <- make_graph("Zachary")
is_directed(G_social)

## 1.9.1. Representacion de la red----
plot(
  G_social,
  vertex.size = 18,
  vertex.color = "lightblue",
  vertex.label.cex = 0.8,
  edge.color = "grey70",
  main = "Red del club de karate de Zachary"
)

## 1.9.2. Orden y tamano de la red----
vcount(G_social)
ecount(G_social)

## 1.9.3. Conexion de la red----
is_connected(G_social)
components(G_social)$no

## 1.9.4. Grado de los vertices----
grados <- degree(G_social)
grados

vertice_mayor_grado <- which.max(grados)
vertice_mayor_grado
grados[vertice_mayor_grado]

data.frame(
  vertice = as_ids(V(G_social)),
  grado = grados
)

sort(grados, decreasing = TRUE)

## 1.9.5. Deteccion de comunidades----
comunidades <- cluster_louvain(G_social)
comunidades

length(comunidades)
sizes(comunidades)
membership(comunidades)

tabla_comunidades <- data.frame(
  vertice = as_ids(V(G_social)),
  comunidad = membership(comunidades),
  grado = grados
)

tabla_comunidades

## 1.9.6. Representacion de las comunidades----
comunidad_vertice <- membership(comunidades)

plot(
  G_social,
  vertex.color = comunidad_vertice,
  vertex.size = 18,
  vertex.label.cex = 0.8,
  edge.color = "grey70",
  main = "Comunidades de la red"
)

plot(
  comunidades,
  G_social,
  vertex.size = 18,
  vertex.label.cex = 0.8,
  edge.color = "grey70",
  main = "Comunidades detectadas mediante Louvain"
)

## 1.9.7. Vertice mas influyente de cada comunidad----
ids_comunidades <- sort(unique(membership(comunidades)))
ids_comunidades

influencers_grado <- sapply(
  ids_comunidades,
  function(i) {
    vertices_comunidad <- V(G_social)[
      membership(comunidades) == i
    ]
    grados_comunidad <- degree(
      G_social,
      v = vertices_comunidad
    )
    as_ids(
      vertices_comunidad[
        which.max(grados_comunidad)
      ]
    )
  }
)

influencers_grado

resultado_influencers <- data.frame(
  comunidad = ids_comunidades,
  influencer = influencers_grado,
  grado = grados[
    match(influencers_grado, as_ids(V(G_social)))
  ]
)

resultado_influencers

## 1.9.8. Estudio de la eliminacion de todos los vertices----
impacto_vertices <- do.call(
  rbind,
  lapply(seq_along(V(G_social)), function(v) {
    
    G_reducido <- delete_vertices(
      G_social,
      v
    )
    
    data.frame(
      vertice = v,
      grado = degree(G_social, v = v),
      sigue_conexo = is_connected(G_reducido),
      numero_componentes = components(G_reducido)$no
    )
  })
)



impacto_vertices

impacto_vertices[
  !impacto_vertices$sigue_conexo,
]

