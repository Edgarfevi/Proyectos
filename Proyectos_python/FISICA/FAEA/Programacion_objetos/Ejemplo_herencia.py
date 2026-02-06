class Animal:
    def GetVelocidad(self):
        '''permite establecer el valor del atributo velocidad
        '''
        return self.velocidad
    
    def SetVelocidad(self,vel):
        '''devuelve la velocidad del animal
        '''
        self.velocidad = vel
        return self.velocidad 
    
    def __init__(self ,  nombre = '' ,  n_patas = 4, peso = 3, velocidad = 11 ):
        self.nombre = nombre
        self.n_patas = n_patas
        self.peso = peso #kg
        self.velocidad = velocidad #m/s    


class Gato(Animal): 
    #La clase Gato hereda de Animal
    def persigue_raton(self,vel_raton):
        '''Calcula si el gato es capaz de atrapar el raton
        '''
        vel_gato = self.GetVelocidad()
        if vel_gato > vel_raton:
            print('%6s atrapa raton'%(self.nombre))
        else: 
            print('raton escapa')
    def acariciar(self):
        print('prrrrr...')
       
    def __init__(self ,nombre ='', velocidad=10, pes=3.4, raza = 'siames', sexo = 'F' ):
        #tomamos el constructor de animal
        Animal.__init__(self,nombre = nombre, peso = pes, n_patas = 4, velocidad =velocidad )
        #definimos nuevos atributos
        self.raza = raza
        self.sexo = sexo
        #fijamos numero de patas...
        self.n_patas = 4 


#Primero probamos la case Animal
Perro = Animal('Pancho', 4, 4, 15)
v_perro = Perro.GetVelocidad() 
print("El perro %6s se mueve a %.1f m/s"%(Perro.nombre, v_perro))
v_perro_2 = Perro.SetVelocidad(10)
print('Modificamos velocidad...')
print("El perro %6s se mueve a %.1f m/s"%(Perro.nombre, v_perro_2))


#Ahora probamos la clase Gato:

Gato_1 = Gato( nombre = 'Milu', raza ='Persa')
velocidad_gato = Gato_1.GetVelocidad()
print(('velocidad gato %s = %0.1f m/s')%(Gato_1.nombre,velocidad_gato))
Gato_1.SetVelocidad(20) #probamos a cambiar si velocidad
velocidad_gato = Gato_1.GetVelocidad()
print(('velocidad gato %s = %0.1f m/s')%(Gato_1.nombre,velocidad_gato))

#pero estos metodos vienen de la clase animal. Ahora usamos el un metodo de la clase gato:
print('Acariciamos...')
Gato_1.acariciar()

#O el metodo persigue raton:

v_raton = 15
Gato_2 = Gato( nombre = 'Lola', velocidad = 10)
print(('%s perisigue raton...')%(Gato_1.nombre))
Gato_1.persigue_raton(v_raton)
print('%s perisigue raton...'%(Gato_2.nombre))
Gato_2.persigue_raton(v_raton)
