# Actualizar R ----
library(installr)
updateR()

# Repaso de R ----

## R como calculadora ----
2+3
5/2
pi
sqrt(4)
5^3
75^(1/3)

## Variables en R ----

a <- 14
a = 7
b <- a*2
e <- 5
d <- a+e
ls() # elementos guardados

# Ejercicio

n <- 10

max_aristas <- n*(n-1)/2
max_arcos <- n*(n-1)

# Funciones en R ----
# Nota: Ctrl+Shift+C para comentar

# Estructura general de una función:
# nombre <- function(argumentos){
#   operaciones
#   return(objeto)
# }

area_rectangulo <- function(lado1=1, lado2=1, ...){
  area <- lado1*lado2
  print(area)
  return(area)
}

area_rectangulo(2,3)
a <- area_rectangulo(lado1=2, lado2=3, 4)

## Instrucciones de control ----
# Estructura general de un condicional
# if(condicion){
#   operaciones
# }

x <- 7

# Si solo se usa una línea, no son necesarias las llaves
if(x>5) cat("x es mayor que 5")
if(x>5){
  cat("x es mayor que 5")
}else{
  cat("x no es mayor que 5")
}

ifelse(x>5, "x es mayor que 5", "x no es mayor que 5")

for(i in 1:4) print(i)
dias <- c("lunes", "martes", "miercoles", "jueves", "viernes")

for(dia in dias) print(dia)

x<-3

while(x<5){
  print(x)
  x <- x+1
}

# Ejercicio

coste_trayecto <- function(distancia, precio_km=0.35, peaje=0){
  
  coste <- distancia*precio_km + peaje
  
  if(distancia > 200){
    coste <- coste*0.9
  }
  
  return(coste)
}

coste_trayecto(150)
coste_trayecto(250, peaje=12)
coste_trayecto(300, precio_km=0.40, peaje=8)

# 17/09/2026
# Vectores ----
a <- c(1,3,5)
a[1]
a[2]
a[1:2]
a[c(1,3)]
1:3
a[-2] # todas las posiciones menos la 2
a[-c(1,3)] # todas las posiciones menos la 1 y 3

b <- c(2,4,6)
a+b
a*b
b[4] <- 8
b
a*b 
# al multiplicar vectores de distinta longitud, 
# opera con normalidad y cuando llega a los elementos extra
# va ciclando -> c(a[1]*b[1], a[2]*b[2], a[3]*b[3], a[1]*b[4])

# Funciones lógicas
a[c(T,T,F)]
a[a<5]
which(a<5)
a[which(a<5)]
max(a)
which.max(a)

x <- c(3,7,4,7,5,6)
max(x)
which.max(x) # solo saca una posición, la del primer máximo que encuentra
which(x == max(x)) # aquí sí devuelve todas las posiciones

numeric(7) # vector de 7 ceros
unique(x) # mismo vector sin valores repetidos

a
x
setdiff(a,x) # vector de elementos de a que no están en x

1:5
5:1
seq(from=1, to=5, by=0.5) # como el np.arange de python
seq(1.5, by=0.5)
seq(1.5, length.out=15)

rep(1, times=5)
rep(1:3, times=5)
rep(1:3, each=5)

# Matrices ----

A <- matrix(1:9); A # El ; es un salto de línea
# Por defecto R almacena todo por columnas
A <- matrix(1:9, nrow=3, byrow=T) # byrow=T hace que sea por filas

rbind(A, 1:3) # añade una fila
cbind(A, 1:3) # añade una columna

# Manera alternativa de crear una matriz
B <- rbind(c(1,3,5),
           c(3,2,4),
           c(4,5,6)
           )
B
B[1,3]
B[1,1:3] 
B[1,] # primera fila
B[,3] # tercera columna

nrow(B) # número de filas
ncol(B) # número de columnas
dim(B) # dimensión

t(B) # matriz transpuesta
B
diag(B) <- 1
B

all(B == t(B)) # devuelve true si es simétrica la matriz
isSymmetric(B) # también se puede usar
A
B
A+B
A*B # Esto multiplica elemento a elemento
A%*%B # Esto es la multiplicación matricial
A^2 # Potencia elemento a elemento
A%*%A # no es cómodo pero por defecto se usa esto para potencias

# Esta librería hace las potencias matriciales más cómodas
library(expm)
A%^%2
all(A%^%2 == A%*%A)

B
which(B == max(B), arr.ind=T)


# Listas ----

lista <- list(1:7, c("palabraa", "palabra"), B, c(T,F))
lista
lista[[2]][2]
lista[[3]][2,3]

# En vez de usar doble corchete, podemos darles nombres
# a los elementos de la lista y acceder a ellos con $
names(lista) <- c("A", "B", "C", "D")
lista$B
lista$C[2,3]

# También se pueden dar nombres en la definición
lista <- list(A=1:7, B= c("palabraa", "palabra"), C=B, D=c(T,F))
lista[[2]]
lista[["B"]]
lista$B


# Data frames ----

datos <- data.frame(
  Grado = c("B", "A", "B"),
  Edad = c(20, 21, 22)
)

mean(datos$Edad)
datos[,2]
datos[1,]
nrow(datos) # n datos medidos
ncol(datos) # n variables que se estudian
datos[,"Edad"]
datos[datos$Edad < 22,]







