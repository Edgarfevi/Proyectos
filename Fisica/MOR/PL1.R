# Práctica del tema 1 ----
# Grafos en R ----
library(igraph)

# Generar un Grafo
G <- make_graph(
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

G
V(G)
E(G)

# Visualización del grafo
plot(G,vertex.size=30,vertex.color="lightblue",vertex.label.size=.8,layout=layout_as_tree(G))

plot(G,vertex.size=30,vertex.color="lightblue",vertex.label.size=.8,layout=layout_as_star(G))

plot(G,vertex.size=30,vertex.color="lightblue",vertex.label.size=.8,layout=layout_with_fr(G))

# Vertice aislado?
Ga <- graph(c(1,2,1,3,2,3),directed = F)

## Opción 1
Ga <- add_vertices ( Ga,1)
plot (Ga)

## Opción 2
Ga_2 = make_graph(c(1,2,1,3,2,3),directed = F,n=4)
plot(Ga_2)

# Cambiar el nombre a los vértices
V(Ga)$name <- c("A","B","C","D")
plot(Ga)

# o(G) = n; vértices
V(G)
vcount(G)

# t(G) = m; aristas
E(G)
ecount(G)

# máxima multiplicidad
any_multiple(G)

!any_multiple(G)

#Añadir ejes
Gbis <- add_edges(G,c(1,2))
plot(Gbis)
any_multiple(Gbis)
count_multiple(Gbis)
max(count_multiple(Gbis)) #Obtener el valor de p

